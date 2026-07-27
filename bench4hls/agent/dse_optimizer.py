"""Intelligent Design Space Exploration — smart pragma selection & optimization strategies."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .diagnoser import DiagnosisResult
from .budget_manager import task_difficulty

logger = logging.getLogger(__name__)

# ── Pragma strategy taxonomy ────────────────────────────────────────

@dataclass
class PragmaStrategy:
    """A specific pragma configuration to try."""
    name: str
    description: str
    pragma_lines: list[str]  # lines to insert (or hints for LLM)
    cost_increase: str = "minimal"  # minimal / moderate / significant
    target_metric: str = "latency"  # latency / area / power / throughput


# ── Error-type → DSE Strategy mappings ─────────────────────────────

FIX_STRATEGIES: dict[str, list[PragmaStrategy]] = {
    "timing": [
        PragmaStrategy(
            name="pipeline_ii1",
            description="Pipeline main loop with II=1",
            pragma_lines=["#pragma HLS PIPELINE II=1"],
            target_metric="latency",
        ),
        PragmaStrategy(
            name="unroll_partial",
            description="Partially unroll loops with factor 2",
            pragma_lines=["#pragma HLS UNROLL factor=2"],
            cost_increase="moderate",
            target_metric="latency",
        ),
        PragmaStrategy(
            name="array_partition_complete",
            description="Complete array partitioning for max bandwidth",
            pragma_lines=["#pragma HLS ARRAY_PARTITION variable=<array> complete dim=1"],
            cost_increase="significant",
            target_metric="latency",
        ),
        PragmaStrategy(
            name="dataflow_task",
            description="Enable task-level parallelism with DATAFLOW",
            pragma_lines=["#pragma HLS DATAFLOW"],
            cost_increase="moderate",
            target_metric="throughput",
        ),
    ],
    "resource": [
        PragmaStrategy(
            name="allocation_dsp",
            description="Limit DSP usage to share resources",
            pragma_lines=["#pragma HLS ALLOCATION instances=mul limit=1"],
            cost_increase="minimal",
            target_metric="area",
        ),
        PragmaStrategy(
            name="reduce_unroll",
            description="Remove or reduce UNROLL factor",
            pragma_lines=["// Remove or reduce #pragma HLS UNROLL"],
            cost_increase="minimal",
            target_metric="area",
        ),
        PragmaStrategy(
            name="bind_op_dsp",
            description="Map multiplications to BRAM instead of LUT",
            pragma_lines=["#pragma HLS BIND_OP variable=<var> op=mul impl=fabric"],
            cost_increase="minimal",
            target_metric="area",
        ),
        PragmaStrategy(
            name="array_reshape",
            description="Reshape arrays for efficient memory access",
            pragma_lines=["#pragma HLS ARRAY_RESHAPE variable=<array> complete dim=1"],
            cost_increase="moderate",
            target_metric="area",
        ),
    ],
    "deadlock": [
        PragmaStrategy(
            name="stream_fifo_depth",
            description="Set FIFO depth for stream to prevent deadlock",
            pragma_lines=["#pragma HLS STREAM variable=<stream> depth=16"],
            cost_increase="minimal",
            target_metric="latency",
        ),
        PragmaStrategy(
            name="dataflow_stream",
            description="Enable DATAFLOW with stream interface for concurrent processing",
            pragma_lines=["#pragma HLS DATAFLOW"],
            cost_increase="moderate",
            target_metric="throughput",
        ),
    ],
    "sim_mismatch": [
        PragmaStrategy(
            name="reset_add",
            description="Add explicit reset for all registers",
            pragma_lines=["#pragma HLS RESET variable=<var>"],
            cost_increase="minimal",
            target_metric="latency",
        ),
        PragmaStrategy(
            name="init_registers",
            description="Initialize all registers with explicit values",
            pragma_lines=["// Initialize all static/register variables explicitly"],
            cost_increase="minimal",
            target_metric="latency",
        ),
    ],
    "synthesis": [
        PragmaStrategy(
            name="inline_manual",
            description="Force inline small functions",
            pragma_lines=["#pragma HLS INLINE"],
            cost_increase="minimal",
            target_metric="latency",
        ),
        PragmaStrategy(
            name="loop_flatten",
            description="Flatten nested loops where possible",
            pragma_lines=["#pragma HLS LOOP_FLATTEN"],
            cost_increase="moderate",
            target_metric="latency",
        ),
    ],
}


# ── Optimization strategies for already-correct code ───────────────

OPTIMIZATION_STRATEGIES: list[PragmaStrategy] = [
    PragmaStrategy("pipeline_ii1", "#pragma HLS PIPELINE II=1", ["#pragma HLS PIPELINE II=1"], "minimal", "latency"),
    PragmaStrategy("unroll_factor8", "#pragma HLS UNROLL factor=8", ["#pragma HLS UNROLL factor=8"], "moderate", "latency"),
    PragmaStrategy("dataflow", "#pragma HLS DATAFLOW", ["#pragma HLS DATAFLOW"], "moderate", "throughput"),
    PragmaStrategy("array_partition_cyclic", "Array partition cyclic dim=1", ["#pragma HLS ARRAY_PARTITION variable=<array> cyclic factor=4 dim=1"], "significant", "latency"),
    PragmaStrategy("loop_merge", "#pragma HLS LOOP_MERGE", ["#pragma HLS LOOP_MERGE"], "minimal", "area"),
]


# ── DSE Engine ──────────────────────────────────────────────────────

@dataclass
class DSEState:
    """Track which strategies have been tried for a task."""
    task_id: str
    tried_strategies: list[str] = field(default_factory=list)
    current_best_latency: Optional[float] = None
    current_best_lut: Optional[float] = None
    current_best_ff: Optional[float] = None


class DSEOptimizer:
    """Intelligent Design Space Explorer — suggests pragma strategies based on context."""

    def __init__(self):
        self._states: dict[str, DSEState] = {}

    # ── public API ─────────────────────────────────────────────────

    def suggest_fix_strategies(
        self,
        task_id: str,
        error_type: str,
        max_suggestions: int = 2,
    ) -> list[PragmaStrategy]:
        """Suggest pragma fixes for a given error type."""
        candidates = FIX_STRATEGIES.get(error_type, [])
        if not candidates:
            return self._default_fix_strategies()

        state = self._get_state(task_id)
        untried = [s for s in candidates if s.name not in state.tried_strategies]

        # Sort: minimal cost first
        cost_order = {"minimal": 0, "moderate": 1, "significant": 2}
        untried.sort(key=lambda s: cost_order.get(s.cost_increase, 0))

        return untried[:max_suggestions]

    def suggest_optimization_strategies(
        self,
        task_id: str,
        current_metrics: dict,
        ref_metrics: Optional[dict] = None,
        max_suggestions: int = 3,
    ) -> list[PragmaStrategy]:
        """Suggest optimization strategies based on PPA gap analysis."""

        state = self._get_state(task_id)
        untried = [s for s in OPTIMIZATION_STRATEGIES if s.name not in state.tried_strategies]

        if not untried:
            return []

        # Priority: target the metric with the largest gap vs reference
        if ref_metrics and current_metrics:
            lat_gap = self._gap(current_metrics, ref_metrics, "latency")
            lut_gap = self._gap(current_metrics, ref_metrics, "lut_util")

            if lat_gap > 0.2:  # >20% worse latency
                untried.sort(key=lambda s: 0 if s.target_metric == "latency" else 1)
            elif lut_gap > 0.1:  # >10% worse area
                untried.sort(key=lambda s: 0 if s.target_metric == "area" else 1)

        cost_order = {"minimal": 0, "moderate": 1, "significant": 2}
        untried.sort(key=lambda s: cost_order.get(s.cost_increase, 0))

        return untried[:max_suggestions]

    def record_attempt(self, task_id: str, strategy_name: str) -> None:
        """Mark a strategy as tried."""
        state = self._get_state(task_id)
        if strategy_name not in state.tried_strategies:
            state.tried_strategies.append(strategy_name)

    def record_metrics(
        self,
        task_id: str,
        latency: Optional[float] = None,
        lut_util: Optional[float] = None,
        ff_util: Optional[float] = None,
    ) -> None:
        """Update best-known metrics."""
        state = self._get_state(task_id)
        if latency is not None:
            if state.current_best_latency is None or latency < state.current_best_latency:
                state.current_best_latency = latency
        if lut_util is not None:
            if state.current_best_lut is None or lut_util < state.current_best_lut:
                state.current_best_lut = lut_util
        if ff_util is not None:
            if state.current_best_ff is None or ff_util < state.current_best_ff:
                state.current_best_ff = ff_util

    def strategies_summary_text(self, strategies: list[PragmaStrategy]) -> str:
        """Generate a human-readable summary of suggested strategies."""
        lines = [
            "### Suggested Pragma Strategy",
            "",
            "Apply the following pragma(s) to the code and re-evaluate:",
        ]
        for i, s in enumerate(strategies, 1):
            lines.append(f"{i}. **{s.name}**: {s.description}")
            lines.append(f"   Pragma: `{s.pragma_lines[0]}`")
            lines.append(f"   Target: {s.target_metric} | Cost: {s.cost_increase}")
        lines.append("")
        return "\n".join(lines)

    # ── internal ───────────────────────────────────────────────────

    def _get_state(self, task_id: str) -> DSEState:
        if task_id not in self._states:
            self._states[task_id] = DSEState(task_id=task_id)
        return self._states[task_id]

    @staticmethod
    def _gap(current: dict, ref: dict, metric: str) -> float:
        """Compute relative gap: (current - ref) / ref. Positive = worse."""
        cv = current.get(metric)
        rv = ref.get(metric)
        if cv is None or rv is None or rv == 0:
            return 1.0
        return (cv - rv) / abs(rv)

    @staticmethod
    def _default_fix_strategies() -> list[PragmaStrategy]:
        """Fallback strategies when error_type is unknown."""
        return [
            PragmaStrategy("inline_all", "Force inline all functions", ["#pragma HLS INLINE"], "minimal", "latency"),
            PragmaStrategy("pipeline_ii1", "Add pipeline pragma", ["#pragma HLS PIPELINE II=1"], "minimal", "latency"),
        ]