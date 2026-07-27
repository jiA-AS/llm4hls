"""Decide next action based on diagnosis, budget, and history."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .diagnoser import DiagnosisResult
from .budget_manager import BudgetManager, task_difficulty

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    RETRY = "retry"           # Fix code and re-evaluate
    OPTIMIZE = "optimize"     # Code passes, but PPA can be improved
    ACCEPT = "accept"         # Code passes and PPA is satisfactory
    SKIP = "skip"             # Give up on this task


@dataclass
class AttemptRecord:
    attempt: int
    code: str
    compilation: str = "N/A"
    simulation: str = "N/A"
    synthesis: str = "N/A"
    error_type: str = ""
    latency: Optional[float] = None
    ff_util: Optional[float] = None
    lut_util: Optional[float] = None
    dsp_util: Optional[float] = None
    bram_util: Optional[float] = None
    power: Optional[float] = None


class DecisionEngine:
    """Make decisions on whether to retry, accept, optimize, or skip a task."""

    def __init__(
        self,
        max_retries_per_type: int = 2,
        ppa_improvement_threshold: float = 0.05,
        ref_metrics: Optional[dict] = None,
    ):
        self.max_retries_per_type = max_retries_per_type
        self.ppa_improvement_threshold = ppa_improvement_threshold
        self.ref_metrics = ref_metrics or {}

    def decide(
        self,
        diagnosis: DiagnosisResult,
        budget: BudgetManager,
        history: list[AttemptRecord],
        task_id: str,
    ) -> DecisionAction:
        """Core decision logic based on diagnosis result, budget, and history."""

        # ── Priority 1: Check budget ──────────────────────────────
        if not budget.can_retry(task_id):
            logger.info("%s: decision=SKIP (budget exhausted)", task_id)
            return DecisionAction.SKIP

        tb = budget.get_or_create(task_id)

        # ── Priority 2: All passed → possibly optimize or accept ──
        if self._all_passed(diagnosis):
            if self._can_optimize(tb, history, task_id):
                logger.info("%s: decision=OPTIMIZE (all passed, room for PPA improvement)", task_id)
                return DecisionAction.OPTIMIZE
            logger.info("%s: decision=ACCEPT (all passed, PPA satisfactory)", task_id)
            return DecisionAction.ACCEPT

        # ── Priority 3: Fail → decide retry strategy ──────────────
        if diagnosis.error_type in ("syntax", "type_mismatch", "interface"):
            # Quick-fix errors: retry aggressively
            same_type_count = sum(
                1 for h in history if h.error_type == diagnosis.error_type
            )
            if same_type_count >= self.max_retries_per_type:
                logger.info("%s: decision=SKIP (%s error repeated %d times)",
                            task_id, diagnosis.error_type, same_type_count)
                return DecisionAction.SKIP
            logger.info("%s: decision=RETRY (%s error, attempt %d/%d)",
                        task_id, diagnosis.error_type,
                        tb.attempts + 1, tb.max_attempts)
            return DecisionAction.RETRY

        if diagnosis.error_type in ("sim_mismatch", "cosim", "deadlock"):
            # Logic/simulation errors: limited retries
            same_type_count = sum(
                1 for h in history if h.error_type in ("sim_mismatch", "cosim", "deadlock")
            )
            if same_type_count >= self.max_retries_per_type + 1:
                logger.info("%s: decision=SKIP (%s error repeated %d times)",
                            task_id, diagnosis.error_type, same_type_count)
                return DecisionAction.SKIP
            logger.info("%s: decision=RETRY (%s error, attempt %d/%d)",
                        task_id, diagnosis.error_type,
                        tb.attempts + 1, tb.max_attempts)
            return DecisionAction.RETRY

        if diagnosis.error_type in ("timing", "resource", "synthesis"):
            # Hardware-specific errors: fewer retries (harder to fix)
            same_type_count = sum(
                1 for h in history if h.error_type in ("timing", "resource", "synthesis")
            )
            if same_type_count >= max(1, self.max_retries_per_type - 1):
                logger.info("%s: decision=SKIP (%s error repeated %d times)",
                            task_id, diagnosis.error_type, same_type_count)
                return DecisionAction.SKIP
            logger.info("%s: decision=RETRY (%s error, attempt %d/%d)",
                        task_id, diagnosis.error_type,
                        tb.attempts + 1, tb.max_attempts)
            return DecisionAction.RETRY

        # Unknown error: one retry, then skip
        unknown_count = sum(1 for h in history if h.error_type == "unknown")
        if unknown_count >= 1:
            logger.info("%s: decision=SKIP (unknown error repeated)", task_id)
            return DecisionAction.SKIP
        logger.info("%s: decision=RETRY (unknown error, attempt %d/%d)",
                    task_id, tb.attempts + 1, tb.max_attempts)
        return DecisionAction.RETRY

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _all_passed(diagnosis: DiagnosisResult) -> bool:
        return (
            diagnosis.compilation == "PASS"
            and diagnosis.simulation == "PASS"
            and diagnosis.synthesis == "PASS"
        )

    def _can_optimize(
        self,
        tb,
        history: list[AttemptRecord],
        task_id: str,
    ) -> bool:
        """Check if optimization is worth attempting."""
        # Only optimize if we haven't already optimized this task
        already_optimized = any(
            h.compilation == "PASS" and h.simulation == "PASS" and h.synthesis == "PASS"
            for h in history[:-1]  # exclude current (latest) attempt
        )
        if already_optimized:
            return False  # Already did a pass → optimize → pass cycle

        # Check if PPA is significantly worse than reference
        if self.ref_metrics and history:
            latest = history[-1]
            if latest.latency is not None and "latency" in self.ref_metrics:
                ref_lat = float(self.ref_metrics["latency"])
                if latest.latency > ref_lat * (1 + self.ppa_improvement_threshold):
                    return True
        return False