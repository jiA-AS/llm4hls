"""Agent main loop: iterative generate → evaluate → diagnose → decide."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..backends.base import ModelBackend
from ..extractor import extract_hls_code, write_design_files
from .diagnoser import Diagnoser, DiagnosisResult
from .decision_engine import DecisionEngine, DecisionAction, AttemptRecord
from .budget_manager import BudgetManager, GlobalBudget, task_difficulty
from .prompt_builder import PromptBuilder, TaskCondition
from .dse_optimizer import DSEOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Mutable state for one task across the Agent loop."""
    task_id: str
    task_spec: str
    condition: str = TaskCondition.CORRECT_UNOPTIMIZED
    starting_code: Optional[str] = None

    attempts: int = 0
    history: list[AttemptRecord] = field(default_factory=list)
    best_code: str = ""
    best_diagnosis: Optional[DiagnosisResult] = None
    final_action: str = "pending"  # accept / skip / exhausted


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    max_total_attempts: int = 500
    csim_budget: int = 500
    synth_budget: int = 500
    token_budget: int = 2_000_000
    max_attempts_per_task: int = 5
    max_retries_per_error_type: int = 2
    ppa_threshold: float = 0.05
    budget_preset: str = "default"  # default / aggressive / conservative


class AgentRunner:
    """Orchestrate the Agent loop over all tasks."""

    def __init__(
        self,
        backend: ModelBackend,
        config: AgentConfig,
        workdir: Path,
        tb_dir: Path,
        tcl_dir: Path,
        input_prompts: Path,
        ref_report: Optional[Path] = None,
        vitis_bin: Optional[str] = None,
        vivado_bin: Optional[str] = None,
        xilinx_version: str = "2025.2.1",
        design_timeout: int = 300,
        parallel_workers: int = 1,
    ):
        self.backend = backend
        self.config = config
        self.workdir = workdir.resolve()
        self.tb_dir = tb_dir.resolve()
        self.tcl_dir = tcl_dir.resolve()
        self.input_prompts = input_prompts
        self.vitis_bin = vitis_bin
        self.vivado_bin = vivado_bin
        self.xilinx_version = xilinx_version
        self.design_timeout = design_timeout
        self.parallel_workers = parallel_workers

        # Sub-modules
        self.diagnoser = Diagnoser(workdir)
        self.decision_engine = DecisionEngine(
            max_retries_per_type=config.max_retries_per_error_type,
            ppa_improvement_threshold=config.ppa_threshold,
        )
        self.dse_optimizer = DSEOptimizer()

        # Budget
        presets = self._choose_presets(config.budget_preset)
        global_budget = GlobalBudget(
            total_csim=config.csim_budget,
            total_synth=config.synth_budget,
            total_tokens=config.token_budget,
        )
        self.budget = BudgetManager(
            global_budget=global_budget,
            presets=presets,
            default_max_attempts=config.max_attempts_per_task,
        )

        # Load tasks
        with open(input_prompts, encoding="utf-8") as f:
            raw = json.load(f)
        self.tasks: dict[str, TaskState] = {}
        for item in raw:
            tid = item.get("task_n", "")
            spec = item.get("input", "")
            if tid:
                self.tasks[tid] = TaskState(task_id=tid, task_spec=spec)

        self.results: list[dict] = []

        # Evaluator (lazy init to avoid loading Xilinx env until needed)
        self._evaluator = None
        self._evaluator_initialized = False

        logger.info("AgentRunner: %d tasks, workdir=%s", len(self.tasks), self.workdir)

    @staticmethod
    def _choose_presets(name: str):
        from .budget_manager import DEFAULT_PRESETS, AGGRESSIVE_PRESETS, CONSERVATIVE_PRESETS
        if name == "aggressive":
            return AGGRESSIVE_PRESETS
        if name == "conservative":
            return CONSERVATIVE_PRESETS
        return DEFAULT_PRESETS

    # ── Main loop ──────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the Agent loop across all tasks. Returns final report dict."""
        task_ids = self.budget.prioritize(list(self.tasks.keys()))
        logger.info("Processing %d tasks (prioritized order)", len(task_ids))

        total_start = time.time()

        for tid in task_ids:
            state = self.tasks[tid]
            logger.info("==== Task %s [%s] ====", tid, task_difficulty(tid))

            try:
                self._process_task(state)
            except Exception as exc:
                logger.error("Task %s: unexpected error: %s", tid, exc)
                state.final_action = "error"

            self.results.append({
                "task_id": state.task_id,
                "attempts": state.attempts,
                "final_action": state.final_action,
                "best_status": {
                    "compilation": state.best_diagnosis.compilation if state.best_diagnosis else "N/A",
                    "simulation": state.best_diagnosis.simulation if state.best_diagnosis else "N/A",
                    "synthesis": state.best_diagnosis.synthesis if state.best_diagnosis else "N/A",
                },
            })

            # Periodic status log
            if (len(self.results) % 10) == 0:
                self._log_progress()

        elapsed = time.time() - total_start

        # Final report
        report = self._build_final_report(elapsed)
        self._log_final_summary(report)

        return report

    def _process_task(self, state: TaskState) -> None:
        """Run the agent loop for a single task."""
        current_code = state.starting_code
        current_prompt = PromptBuilder.build_initial_prompt(
            state.task_spec, state.condition, current_code
        )

        while self.budget.can_retry(state.task_id):
            # 1. Generate
            logger.info("  [%s] attempt %d: generating...", state.task_id, state.attempts + 1)
            t0 = time.time()
            response = self.backend.generate(current_prompt)
            gen_time = time.time() - t0
            code = extract_hls_code(response)
            if not code:
                logger.warning("  [%s] empty extraction — treating as syntax error", state.task_id)
                code = "// EXTRACTION FAILED\n"

            # Count tokens (approximate: 4 chars ≈ 1 token)
            token_count = len(current_prompt + response) // 4
            self.budget.consume_tokens(state.task_id, token_count)
            self.budget.record_attempt(state.task_id)
            state.attempts += 1

            logger.info("  [%s] generated %d chars in %.1fs (~%d tokens)",
                        state.task_id, len(response), gen_time, token_count)

            # 2. Evaluate
            diagnosis = self._evaluate_single(state.task_id, code, state.attempts)

            # 3. Record history
            record = AttemptRecord(
                attempt=state.attempts,
                code=code,
                compilation=diagnosis.compilation,
                simulation=diagnosis.simulation,
                synthesis=diagnosis.synthesis,
                error_type=diagnosis.error_type,
                latency=diagnosis.latency,
                ff_util=diagnosis.ff_util,
                lut_util=diagnosis.lut_util,
                dsp_util=diagnosis.dsp_util,
                bram_util=diagnosis.bram_util,
                power=diagnosis.power,
            )
            state.history.append(record)

            # Track best
            if self._is_better_than_current(diagnosis, state.best_diagnosis):
                state.best_code = code
                state.best_diagnosis = diagnosis

            # 4. Decide
            action = self.decision_engine.decide(
                diagnosis, self.budget, state.history, state.task_id,
            )

            if action == DecisionAction.ACCEPT:
                state.final_action = "accept"
                return
            elif action == DecisionAction.SKIP:
                state.final_action = "skip"
                return
            elif action == DecisionAction.OPTIMIZE:
                # Use DSE to suggest optimization strategies
                strategies = self.dse_optimizer.suggest_optimization_strategies(
                    state.task_id,
                    current_metrics={
                        "latency": diagnosis.latency,
                        "ff_util": diagnosis.ff_util,
                        "lut_util": diagnosis.lut_util,
                        "dsp_util": diagnosis.dsp_util,
                        "bram_util": diagnosis.bram_util,
                        "power": diagnosis.power,
                    },
                )
                dse_text = self.dse_optimizer.strategies_summary_text(strategies) if strategies else ""
                current_prompt = PromptBuilder.build_optimize_prompt(
                    code,
                    current_metrics={
                        "latency": diagnosis.latency,
                        "ff_util": diagnosis.ff_util,
                        "lut_util": diagnosis.lut_util,
                        "dsp_util": diagnosis.dsp_util,
                        "bram_util": diagnosis.bram_util,
                        "power": diagnosis.power,
                    },
                )
                # Append DSE suggestions to the prompt
                if dse_text:
                    current_prompt += "\n" + dse_text
                # Record the strategy as tried
                for s in strategies:
                    self.dse_optimizer.record_attempt(state.task_id, s.name)
            else:  # RETRY
                # Use DSE to suggest fix strategies based on error type
                strategies = self.dse_optimizer.suggest_fix_strategies(
                    state.task_id, diagnosis.error_type
                )
                dse_text = self.dse_optimizer.strategies_summary_text(strategies) if strategies else ""
                current_prompt = PromptBuilder.build_fix_prompt(code, diagnosis)
                if dse_text:
                    current_prompt += "\n" + dse_text
                for s in strategies:
                    self.dse_optimizer.record_attempt(state.task_id, s.name)

        state.final_action = "exhausted"

    # ── Evaluation ─────────────────────────────────────────────────

    def _evaluate_single(self, task_id: str, code: str, run: int) -> DiagnosisResult:
        """Run Bench4HLS-style evaluation for a single design."""
        # Write design file
        design_name = f"{task_id}_design_run{run}.cpp"
        design_path = self.workdir / design_name
        design_path.write_text(code, encoding="utf-8")

        # Copy testbench (once)
        tb_name = f"{task_id}_tb.cpp"
        tb_dest = self.workdir / tb_name
        if not tb_dest.exists():
            tb_src = self.tb_dir / tb_name
            if tb_src.exists():
                tb_dest.write_text(tb_src.read_text(encoding="utf-8"), encoding="utf-8")

        # Init evaluator lazily
        self._ensure_evaluator()

        if not self._evaluator_initialized:
            return DiagnosisResult(
                task_id=task_id, run=run,
                compilation="SKIP", simulation="SKIP", synthesis="SKIP",
                error_type="unknown",
                error_summary="Evaluator not available (Xilinx tools missing)",
            )

        # Run simulation and synthesis via Bench4HLS evaluator
        try:
            sim_ok = self._evaluator._sim_one(1, run, self._get_prob_num(task_id), task_id)
            synth_ok = self._evaluator._synth_one(1, run, self._get_prob_num(task_id), task_id)

            self.budget.consume_csim(task_id)
            self.budget.consume_synth(task_id)
        except Exception as exc:
            logger.error("  [%s] evaluation exception: %s", task_id, exc)
            return DiagnosisResult(
                task_id=task_id, run=run,
                compilation="FAIL", simulation="FAIL", synthesis="FAIL",
                error_type="unknown",
                error_summary=str(exc)[:200],
            )

        # Run Vivado power (if available)
        if self.vivado_bin and self._evaluator_initialized:
            try:
                self._evaluator._vivado_one(run, task_id)
            except Exception:
                pass

        # Diagnose
        return self.diagnoser.diagnose(task_id, run)

    @staticmethod
    def _get_prob_num(task_id: str) -> int:
        import re
        m = re.search(r"(\d+)", task_id)
        return int(m.group(1)) if m else 1

    def _ensure_evaluator(self) -> None:
        if self._evaluator is not None:
            return
        if self.vitis_bin is None:
            logger.warning("Vitis/Vivado not configured — skipping hardware evaluation")
            self._evaluator_initialized = False
            return
        try:
            from ..evaluator import Evaluator
            self._evaluator = Evaluator(
                workdir=self.workdir,
                scripts_dir=self.tcl_dir,
                version=self.xilinx_version,
                vitis_bin=self.vitis_bin,
                vivado_bin=self.vivado_bin,
                skip_power=False,
                parallel_workers=self.parallel_workers,
                design_timeout_seconds=self.design_timeout,
            )
            self._evaluator_initialized = True
            logger.info("Evaluator initialized")
        except Exception as exc:
            logger.warning("Failed to initialize Evaluator: %s", exc)
            self._evaluator_initialized = False

    # ── Best selection ─────────────────────────────────────────────

    @staticmethod
    def _is_better_than_current(
        new: DiagnosisResult,
        current: Optional[DiagnosisResult],
    ) -> bool:
        if current is None:
            return True
        # All-pass is always better than any failure
        new_ok = new.compilation == "PASS" and new.simulation == "PASS" and new.synthesis == "PASS"
        cur_ok = current.compilation == "PASS" and current.simulation == "PASS" and current.synthesis == "PASS"
        if new_ok and not cur_ok:
            return True
        if not new_ok and cur_ok:
            return False
        # Both pass or both fail: prefer lower latency
        if new.latency is not None and current.latency is not None:
            return new.latency < current.latency
        return False

    # ── Reporting ──────────────────────────────────────────────────

    def _log_progress(self) -> None:
        stats = self.budget.stats()
        logger.info(
            "PROGRESS: %d/%d tasks, %d attempts, csim=%d/%d synth=%d/%d tokens=%d/%d",
            stats["completed"], stats["total_tasks"],
            stats["total_attempts"],
            stats["csim_used"], stats["csim_remaining"] + stats["csim_used"],
            stats["synth_used"], stats["synth_remaining"] + stats["synth_used"],
            stats["tokens_used"], stats["tokens_remaining"] + stats["tokens_used"],
        )

    def _build_final_report(self, elapsed: float) -> dict:
        stats = self.budget.stats()
        accepted = sum(1 for r in self.results if r["final_action"] == "accept")
        skipped = sum(1 for r in self.results if r["final_action"] == "skip")
        exhausted = sum(1 for r in self.results if r["final_action"] == "exhausted")
        errors = sum(1 for r in self.results if r["final_action"] == "error")

        return {
            "total_tasks": len(self.results),
            "accepted": accepted,
            "skipped": skipped,
            "exhausted": exhausted,
            "errors": errors,
            "total_attempts": stats["total_attempts"],
            "csim_used": stats["csim_used"],
            "synth_used": stats["synth_used"],
            "tokens_used": stats["tokens_used"],
            "elapsed_seconds": elapsed,
            "per_task": self.results,
        }

    def _log_final_summary(self, report: dict) -> None:
        logger.info("=" * 55)
        logger.info("  AGENT RUN COMPLETE")
        logger.info("=" * 55)
        logger.info("  Tasks:        %d total", report["total_tasks"])
        logger.info("  Accepted:     %d", report["accepted"])
        logger.info("  Skipped:      %d", report["skipped"])
        logger.info("  Exhausted:    %d", report["exhausted"])
        logger.info("  Errors:       %d", report["errors"])
        logger.info("  Attempts:     %d", report["total_attempts"])
        logger.info("  CSIM calls:   %d", report["csim_used"])
        logger.info("  Synth calls:  %d", report["synth_used"])
        logger.info("  Tokens used:  %d", report["tokens_used"])
        logger.info("  Elapsed:      %.1f s", report["elapsed_seconds"])
        logger.info("=" * 55)

        # Save report
        report_path = self.workdir / "agent_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Report saved → %s", report_path)