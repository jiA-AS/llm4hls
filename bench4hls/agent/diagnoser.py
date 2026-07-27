"""Parse sim_out/ synth_out/ files and diagnose failure reasons."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Error-type taxonomy ────────────────────────────────────────────
_ERROR_PATTERNS: dict[str, list[str]] = {
    "syntax": [
        r"error:\s*syntax",
        r"error:\s*expected",
        r"undeclared\s+identifier",
        r"unknown\s+type\s+name",
        r"no\s+matching\s+function",
        r"use\s+of\s+undeclared",
        r"fatal\s+error",
        r"compilation\s+error",
    ],
    "type_mismatch": [
        r"cannot\s+convert",
        r"invalid\s+conversion",
        r"type\s+mismatch",
        r"no\s+known\s+conversion",
    ],
    "interface": [
        r"top-level\s+function\s+not\s+found",
        r"cannot\s+find\s+top\s+function",
        r"top\s+function\s+.*not\s+found",
    ],
    "timing": [
        r"timing\s+violation",
        r"negative\s+slack",
        r"setup\s+time\s+violation",
        r"II\s+violation",
        r"initiation\s+interval",
        r"pipeline\s+II",
    ],
    "resource": [
        r"resource\s+limit\s+exceeded",
        r"overutilized",
        r"cannot\s+place",
        r"routing\s+congestion",
    ],
    "sim_mismatch": [
        r"mismatch",
        r"assertion\s+failed",
        r"expected.*got",
        r"ERROR\s+at\s+time",
        r"test\s+failed",
    ],
    "deadlock": [
        r"deadlock",
        r"livelock",
        r"stall",
        r"hls::stream.*empty",
        r"stream\s+read\s+timeout",
    ],
}

_COSIM_PATTERNS = [
    r"cosim.*fail",
    r"co-simulation.*fail",
    r"verilog.*simulation.*fail",
    r"RTL.*simulation.*mismatch",
]


@dataclass
class DiagnosisResult:
    """Structured diagnosis for one design evaluation."""

    task_id: str
    run: int
    compilation: str = "N/A"  # PASS / FAIL
    simulation: str = "N/A"  # PASS / FAIL / TIMEOUT
    synthesis: str = "N/A"  # PASS / FAIL
    # -- classified --
    error_type: str = ""  # syntax / type_mismatch / interface / timing / resource / sim_mismatch / deadlock / cosim / unknown
    error_summary: str = ""  # first meaningful error line
    error_detail: str = ""  # full log tail (≤500 chars)
    # -- metrics (populated when synthesis passes) --
    latency: Optional[float] = None
    ff_util: Optional[float] = None
    lut_util: Optional[float] = None
    dsp_util: Optional[float] = None
    bram_util: Optional[float] = None
    power: Optional[float] = None
    # -- raw --
    sim_log_path: Optional[Path] = None
    synth_log_path: Optional[Path] = None
    vivado_log_path: Optional[Path] = None


class Diagnoser:
    """Parse Bench4HLS evaluation outputs and produce DiagnosisResult."""

    def __init__(self, workdir: Path, logs_dir: Optional[Path] = None):
        self.workdir = workdir.resolve()
        self.logs_dir = (logs_dir or self.workdir / "logs").resolve()

    # ── public API ─────────────────────────────────────────────────

    def diagnose(self, task_id: str, run: int) -> DiagnosisResult:
        """Run full diagnosis for a single design."""
        result = DiagnosisResult(task_id=task_id, run=run)

        # 1. Parse sim_out
        sim_file = self.workdir / "sim_out" / f"{task_id}_run{run}.txt"
        result.sim_log_path = sim_file
        if sim_file.exists():
            self._parse_sim_output(sim_file, result)

        # 2. Parse synth_out
        synth_file = self.workdir / "synth_out" / f"{task_id}_run{run}.txt"
        result.synth_log_path = synth_file
        if synth_file.exists():
            self._parse_synth_output(synth_file, result)

        # 3. Parse power_out
        power_file = self.workdir / "power_out" / f"{task_id}_run{run}.txt"
        result.vivado_log_path = power_file
        if power_file.exists():
            self._parse_power_output(power_file, result)

        # 4. If still failing, check Vitis logs for deeper diagnosis
        if result.compilation == "FAIL" or result.simulation == "FAIL" or result.synthesis == "FAIL":
            self._check_vitis_log(task_id, run, result)

        # 5. Classify error type
        if result.compilation == "FAIL" or result.simulation == "FAIL" or result.synthesis == "FAIL":
            self._classify_error(result)

        return result

    # ── internal parsers ───────────────────────────────────────────

    def _parse_sim_output(self, path: Path, result: DiagnosisResult) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        if len(lines) >= 1:
            result.compilation = lines[0].strip()
        if len(lines) >= 2:
            result.simulation = lines[1].strip()
        if len(lines) >= 3:
            result.error_summary = lines[2].strip()
            result.error_detail = "\n".join(lines[2:])[:500]

    def _parse_synth_output(self, path: Path, result: DiagnosisResult) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        if len(lines) >= 1:
            result.synthesis = lines[0].strip()
        try:
            if len(lines) >= 2 and lines[1].strip() not in ("N/A", "FAIL", ""):
                result.latency = float(lines[1].strip())
            if len(lines) >= 3 and lines[2].strip() not in ("N/A", "FAIL", ""):
                result.ff_util = float(lines[2].strip())
            if len(lines) >= 4 and lines[3].strip() not in ("N/A", "FAIL", ""):
                result.lut_util = float(lines[3].strip())
            if len(lines) >= 5 and lines[4].strip() not in ("N/A", "FAIL", ""):
                result.dsp_util = float(lines[4].strip())
            if len(lines) >= 6 and lines[5].strip() not in ("N/A", "FAIL", ""):
                result.bram_util = float(lines[5].strip())
        except (ValueError, IndexError):
            pass

    def _parse_power_output(self, path: Path, result: DiagnosisResult) -> None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            return
        if raw and raw not in ("N/A", "ERROR", ""):
            try:
                result.power = float(raw)
            except ValueError:
                pass

    def _check_vitis_log(self, task_id: str, run: int, result: DiagnosisResult) -> None:
        """Try to read Vitis simulation/synthesis logs for more detail."""
        candidates = [
            self.logs_dir / f"sim_run{run}_design{self._extract_prob_num(task_id)}.log",
            self.logs_dir / f"synth_run{run}_design{self._extract_prob_num(task_id)}.log",
        ]
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                text = cand.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                lower = line.lower()
                # Check for ERROR/WARNING lines first
                if "error" in lower or "warning" in lower or "fail" in lower:
                    if not result.error_summary:
                        result.error_summary = line.strip()[:200]
                    result.error_detail = (result.error_detail + "\n" + line.strip())[:500]
                # Then check against known patterns
                for category, patterns in _ERROR_PATTERNS.items():
                    if category == "sim_mismatch":
                        continue
                    for pat in patterns:
                        if re.search(pat, lower):
                            if not result.error_summary:
                                result.error_summary = line.strip()[:200]
                            result.error_detail = (result.error_detail + "\n" + line.strip())[:500]
                            return
            # If we found error lines but no pattern match, still return (don't skip)
            if result.error_summary:
                return

    @staticmethod
    def _extract_prob_num(task_id: str) -> int:
        try:
            return int(re.search(r"(\d+)", task_id).group(1))
        except Exception:
            return 1

    # ── error classification ──────────────────────────────────────

    def _classify_error(self, result: DiagnosisResult) -> None:
        """Determine error_type from accumulated messages."""
        combined = " ".join(
            filter(None, [
                result.error_summary,
                result.error_detail,
                result.compilation,
                result.simulation,
                result.synthesis,
            ])
        ).lower()

        # Check cosim patterns first
        for pat in _COSIM_PATTERNS:
            if re.search(pat, combined):
                result.error_type = "cosim"
                return

        # Order matters: specific → general
        for category, patterns in _ERROR_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, combined):
                    result.error_type = category
                    return

        # Fallback categorization based on status flags
        if result.synthesis == "FAIL" and result.compilation == "PASS":
            result.error_type = "synthesis"
        elif result.simulation in ("FAIL", "TIMEOUT"):
            result.error_type = "sim_mismatch"
        elif result.compilation == "FAIL":
            result.error_type = "syntax"
        else:
            result.error_type = "unknown"