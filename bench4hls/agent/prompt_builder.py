"""Build structured prompts for initial generation, error fixing, and PPA optimization."""
from __future__ import annotations

from typing import Optional
from .diagnoser import DiagnosisResult

# ── System prompt for the Agent ─────────────────────────────────────
AGENT_SYSTEM_PROMPT = (
    "You are an expert HLS C++ developer participating in an iterative code-generation challenge. "
    "You will receive HLS task specifications and, optionally, compiler/simulation/synthesis feedback "
    "from previous attempts. Your job is to generate synthesizable HLS C++ code that:\n"
    "1. Compiles without errors.\n"
    "2. Passes C simulation (functional correctness).\n"
    "3. Passes HLS synthesis with competitive PPA (Performance, Power, Area).\n\n"
    "Guidelines:\n"
    "- Output ONLY the C++ code (no explanations outside code fences).\n"
    "- Include necessary headers (ap_int.h, ap_fixed.h, hls_stream.h, etc.).\n"
    "- The top-level function MUST be named TopModule and match the given prototype exactly.\n"
    "- Add appropriate HLS pragmas (PIPELINE, UNROLL, ARRAY_PARTITION, etc.).\n"
    "- If fixing errors: only modify the minimal necessary parts, keep the rest intact.\n"
    "- If optimizing PPA: add/improve pragmas, restructure loops, but preserve correctness.\n"
)

# ── Task initial-conditions ─────────────────────────────────────────

class TaskCondition:
    """Represents the 5 initial conditions per FPT 2026 Track A."""
    CORRECT_UNOPTIMIZED = "correct_unoptimized"       # ① 功能正确但未优化
    COMPILE_SYNTH_FAIL = "compile_synth_fail"          # ② 编译/综合失败
    SIM_COSIM_FAIL = "sim_cosim_fail"                  # ③ 仿真/联合仿真失败
    DEADLOCK_STREAM = "deadlock_stream"                # ④ 存在死锁/流行为问题
    OTHER_COMPILE_ISSUE = "other_compile_issue"        # ⑤ 其他编译问题


class PromptBuilder:
    """Construct task-specific prompts for each phase of the Agent loop."""

    @staticmethod
    def build_initial_prompt(
        task_spec: str,
        condition: str = TaskCondition.CORRECT_UNOPTIMIZED,
        starting_code: Optional[str] = None,
    ) -> str:
        """
        Build the initial prompt for a task.

        Args:
            task_spec: The original task description from input_prompts.json
            condition: One of TaskCondition values
            starting_code: Pre-existing code (for fix/optimize conditions)
        """
        if condition == TaskCondition.CORRECT_UNOPTIMIZED and not starting_code:
            return PromptBuilder._build_from_spec_prompt(task_spec)

        return PromptBuilder._build_fix_initial_prompt(task_spec, condition, starting_code)

    @staticmethod
    def build_fix_prompt(
        original_code: str,
        diagnosis: DiagnosisResult,
    ) -> str:
        """Build a repair prompt based on evaluation feedback."""
        error_context = PromptBuilder._format_diagnosis(diagnosis)
        fix_strategy = PromptBuilder._suggest_fix_strategy(diagnosis.error_type)

        return f"""The following HLS C++ code was evaluated and FAILED.

{error_context}

--- ORIGINAL CODE ---
{original_code}

--- FIX INSTRUCTIONS ---
{fix_strategy}

Please provide the COMPLETE corrected code below:
"""

    @staticmethod
    def build_optimize_prompt(
        code: str,
        current_metrics: dict,
        ref_metrics: Optional[dict] = None,
    ) -> str:
        """Build a PPA optimization prompt."""
        lines = [
            "The following HLS C++ code compiles and simulates correctly, but PPA can be improved.",
            "",
            "--- CURRENT METRICS ---",
            f"  Latency:   {current_metrics.get('latency', 'N/A')} ns",
            f"  FF Util:   {current_metrics.get('ff_util', 'N/A')}%",
            f"  LUT Util:  {current_metrics.get('lut_util', 'N/A')}%",
            f"  DSP Util:  {current_metrics.get('dsp_util', 'N/A')}%",
            f"  BRAM Util: {current_metrics.get('bram_util', 'N/A')}%",
            f"  Power:     {current_metrics.get('power', 'N/A')} W",
        ]

        if ref_metrics:
            lines.append("")
            lines.append("--- REFERENCE METRICS (target) ---")
            for k, v in ref_metrics.items():
                lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("--- OPTIMIZATION INSTRUCTIONS ---")
        lines.append(
            "Improve PPA by adding/changing HLS pragmas. Common techniques:\n"
            "- #pragma HLS PIPELINE II=1  (reduce initiation interval)\n"
            "- #pragma HLS UNROLL factor=N  (parallelize loops)\n"
            "- #pragma HLS ARRAY_PARTITION variable=X complete dim=1  (memory bandwidth)\n"
            "- #pragma HLS DATAFLOW  (task-level parallelism)\n"
            "- #pragma HLS ALLOCATION instances=mul limit=N  (resource sharing)\n"
            "Keep the code functionally identical. Output the complete optimized code."
        )
        lines.append("")
        lines.append("--- CURRENT CODE ---")
        lines.append(code)

        return "\n".join(lines)

    # ── internal helpers ───────────────────────────────────────────

    @staticmethod
    def _build_from_spec_prompt(task_spec: str) -> str:
        return (
            "Write synthesizable HLS C++ code for the following task.\n\n"
            f"### Task:\n{task_spec}\n\n"
            "### Requirements:\n"
            "- Include necessary headers (e.g., ap_int.h, ap_fixed.h, hls_stream.h)\n"
            "- Use #pragma HLS INTERFACE for all ports\n"
            "- The top-level function MUST be named TopModule\n"
            "- Output ONLY the C++ code inside ```cpp ... ``` fences\n"
            "- Do NOT add any explanation outside the code fences\n"
        )

    @staticmethod
    def _build_fix_initial_prompt(
        task_spec: str,
        condition: str,
        starting_code: Optional[str],
    ) -> str:
        condition_desc = {
            TaskCondition.CORRECT_UNOPTIMIZED: "The following code is functionally correct but lacks optimization pragmas.",
            TaskCondition.COMPILE_SYNTH_FAIL: "The following code fails compilation or HLS synthesis.",
            TaskCondition.SIM_COSIM_FAIL: "The following code compiles but fails simulation or co-simulation.",
            TaskCondition.DEADLOCK_STREAM: "The following code has deadlock or streaming behavior issues.",
            TaskCondition.OTHER_COMPILE_ISSUE: "The following code has compilation issues.",
        }

        lines = [
            condition_desc.get(condition, "Fix the following HLS C++ code."),
            "",
            f"### Original Task:\n{task_spec}",
        ]

        if starting_code:
            lines.append(f"\n### Current Code:\n{starting_code}")
            lines.append("\n### Instructions:")
            if condition == TaskCondition.CORRECT_UNOPTIMIZED:
                lines.append(
                    "Add HLS pragmas to optimize PPA. Keep functionality unchanged."
                )
            else:
                lines.append(
                    "Fix the code so it compiles, simulates correctly, and synthesizes successfully. "
                    "Output the complete corrected code."
                )
        else:
            lines.append("\nGenerate synthesizable HLS C++ code from scratch.")

        return "\n".join(lines)

    @staticmethod
    def _format_diagnosis(diagnosis: DiagnosisResult) -> str:
        parts = [
            "### EVALUATION RESULTS",
            f"  Task: {diagnosis.task_id} (run {diagnosis.run})",
            f"  Compilation: {diagnosis.compilation}",
            f"  Simulation:  {diagnosis.simulation}",
            f"  Synthesis:   {diagnosis.synthesis}",
            f"  Error Type:  {diagnosis.error_type}",
        ]
        if diagnosis.error_summary:
            parts.append(f"  Error Summary: {diagnosis.error_summary}")
        if diagnosis.error_detail:
            parts.append(f"  Error Detail: {diagnosis.error_detail[:300]}")
        return "\n".join(parts)

    @staticmethod
    def _suggest_fix_strategy(error_type: str) -> str:
        strategies = {
            "syntax": (
                "There is a syntax/compilation error in the code. "
                "Check for missing semicolons, undeclared variables, incorrect types, "
                "or missing #include directives. Fix the compilation errors."
            ),
            "type_mismatch": (
                "There is a type mismatch in the code. "
                "Ensure all variable types match their usage. "
                "Check ap_int/ap_uint widths and signed/unsigned conversions."
            ),
            "interface": (
                "The top-level function interface does not match the required prototype. "
                "Ensure the function is named `TopModule` exactly and has the correct "
                "parameter types, order, and direction (by-reference for outputs)."
            ),
            "timing": (
                "The design failed timing or has pipeline II violations. "
                "Add #pragma HLS PIPELINE II=1 to the main loop. "
                "Consider array partitioning to improve memory bandwidth. "
                "Reduce loop-carried dependencies."
            ),
            "resource": (
                "The design exceeds resource limits. "
                "Add #pragma HLS ALLOCATION to share DSP/BRAM instances. "
                "Reduce unroll factors. Consider resource-sharing across operations."
            ),
            "sim_mismatch": (
                "The simulation output does not match expected values. "
                "Check the algorithm logic carefully. Verify initialization values, "
                "loop bounds, and edge cases. Ensure all outputs are driven correctly."
            ),
            "deadlock": (
                "The design has deadlock or streaming issues. "
                "Ensure hls::stream reads and writes are balanced. "
                "Check that all stream paths eventually consume/produce data. "
                "Add FIFO depth pragmas if needed."
            ),
            "cosim": (
                "Co-simulation (RTL simulation) failed while C simulation passed. "
                "This usually indicates timing-dependent behavior or uninitialized signals. "
                "Add #pragma HLS RESET variable=... for explicit reset behavior. "
                "Ensure all registers have defined reset values."
            ),
            "synthesis": (
                "HLS synthesis failed. "
                "Check for unsupported constructs in synthesis (e.g., dynamic memory allocation, "
                "recursion, floating-point without proper mapping). "
                "Simplify complex expressions, break large loops into smaller sub-blocks."
            ),
            "unknown": (
                "The code failed evaluation for an unspecified reason. "
                "Review the whole code for potential issues: correct includes, "
                "proper function signature, synthesizable constructs only, "
                "no undefined behavior."
            ),
        }
        return strategies.get(error_type, strategies["unknown"])