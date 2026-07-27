"""Track and enforce tool-call and token budgets per task and globally."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Difficulty tiers ──────────────────────────────────────────────

_DIFFICULTY_THRESHOLDS = [
    (50, "easy"),
    (110, "medium"),
    (150, "hard"),
    (170, "expert"),
]


def task_difficulty(task_id: str) -> str:
    """Map a task_id (e.g. 'Prob042') to a difficulty tier."""
    import re
    m = re.search(r"(\d+)", task_id)
    num = int(m.group(1)) if m else 1
    for threshold, tier in _DIFFICULTY_THRESHOLDS:
        if num <= threshold:
            return tier
    return "expert"


@dataclass
class TaskBudget:
    max_csim: int = 3
    max_synth: int = 3
    max_attempts: int = 5
    max_tokens: int = 15000

    csim_used: int = 0
    synth_used: int = 0
    attempts: int = 0
    tokens_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    @property
    def csim_remaining(self) -> int:
        return max(0, self.max_csim - self.csim_used)

    @property
    def synth_remaining(self) -> int:
        return max(0, self.max_synth - self.synth_used)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)


@dataclass
class GlobalBudget:
    total_csim: int = 500
    total_synth: int = 500
    total_tokens: int = 2_000_000
    total_time_seconds: int = 7200

    csim_used: int = 0
    synth_used: int = 0
    tokens_used: int = 0

    @property
    def csim_remaining(self) -> int:
        return max(0, self.total_csim - self.csim_used)

    @property
    def synth_remaining(self) -> int:
        return max(0, self.total_synth - self.synth_used)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.total_tokens - self.tokens_used)


def _make_presets(e, e2, e3, e4, m, m2, m3, m4, h, h2, h3, h4, x, x2, x3, x4):
    return {
        "easy": TaskBudget(max_csim=e, max_synth=e2, max_attempts=e3, max_tokens=e4),
        "medium": TaskBudget(max_csim=m, max_synth=m2, max_attempts=m3, max_tokens=m4),
        "hard": TaskBudget(max_csim=h, max_synth=h2, max_attempts=h3, max_tokens=h4),
        "expert": TaskBudget(max_csim=x, max_synth=x2, max_attempts=x3, max_tokens=x4),
    }


DEFAULT_PRESETS = _make_presets(1, 1, 2, 5000, 2, 2, 4, 12000, 3, 3, 5, 20000, 4, 4, 6, 30000)
AGGRESSIVE_PRESETS = _make_presets(2, 1, 3, 8000, 3, 2, 5, 18000, 4, 3, 6, 28000, 5, 4, 8, 40000)
CONSERVATIVE_PRESETS = _make_presets(1, 1, 1, 3000, 1, 1, 2, 8000, 2, 2, 3, 15000, 2, 2, 4, 20000)


class BudgetManager:
    def __init__(
        self,
        global_budget: Optional[GlobalBudget] = None,
        presets: Optional[dict[str, TaskBudget]] = None,
        default_max_attempts: Optional[int] = None,
    ):
        self.gbudget = global_budget or GlobalBudget()
        self.presets = presets or DEFAULT_PRESETS
        self.default_max_attempts = default_max_attempts
        self._task_budgets: dict[str, TaskBudget] = {}

        logger.info(
            "BudgetManager: global_csim=%d global_synth=%d global_tokens=%d default_max_attempts=%s",
            self.gbudget.total_csim, self.gbudget.total_synth, self.gbudget.total_tokens,
            str(self.default_max_attempts),
        )

    def get_or_create(self, task_id: str) -> TaskBudget:
        if task_id not in self._task_budgets:
            tier = task_difficulty(task_id)
            preset = self.presets.get(tier, self.presets["easy"])
            max_attempts = preset.max_attempts
            if self.default_max_attempts is not None:
                max_attempts = max(1, int(self.default_max_attempts))

            self._task_budgets[task_id] = TaskBudget(
                max_csim=preset.max_csim, max_synth=preset.max_synth,
                max_attempts=max_attempts, max_tokens=preset.max_tokens,
            )
            logger.debug("Task %s [%s]: csim=%d synth=%d attempts=%d tokens=%d",
                         task_id, tier, preset.max_csim, preset.max_synth,
                         preset.max_attempts, preset.max_tokens)
        return self._task_budgets[task_id]

    def can_retry(self, task_id: str) -> bool:
        tb = self.get_or_create(task_id)
        if tb.exhausted:
            logger.debug(
                "Task %s cannot retry: attempts exhausted (%d/%d)",
                task_id, tb.attempts, tb.max_attempts,
            )
            return False

        # If both csim & synth budgets are 0, it means skip-eval mode — allow generation-only retries
        skip_eval = (self.gbudget.total_csim == 0 and self.gbudget.total_synth == 0)
        if not skip_eval:
            if self.gbudget.csim_remaining <= 0 or self.gbudget.synth_remaining <= 0:
                logger.debug(
                    "Task %s cannot retry: global eval budget exhausted (csim_remaining=%d synth_remaining=%d)",
                    task_id, self.gbudget.csim_remaining, self.gbudget.synth_remaining,
                )
                return False

        if self.gbudget.tokens_remaining < 500:
            logger.debug(
                "Task %s cannot retry: global token budget too low (%d < 500)",
                task_id, self.gbudget.tokens_remaining,
            )
            return False

        return True

    def consume_csim(self, task_id: str) -> None:
        self.get_or_create(task_id).csim_used += 1
        self.gbudget.csim_used += 1

    def consume_synth(self, task_id: str) -> None:
        self.get_or_create(task_id).synth_used += 1
        self.gbudget.synth_used += 1

    def consume_tokens(self, task_id: str, count: int) -> None:
        self.get_or_create(task_id).tokens_used += count
        self.gbudget.tokens_used += count

    def record_attempt(self, task_id: str) -> None:
        self.get_or_create(task_id).attempts += 1

    def summary(self, task_id: Optional[str] = None) -> dict:
        g = self.gbudget
        base = {
            "global_csim_remaining": g.csim_remaining,
            "global_synth_remaining": g.synth_remaining,
            "global_tokens_remaining": g.tokens_remaining,
        }
        if task_id:
            tb = self.get_or_create(task_id)
            base.update({
                "task_attempts": tb.attempts,
                "task_max_attempts": tb.max_attempts,
                "task_csim_remaining": tb.csim_remaining,
                "task_synth_remaining": tb.synth_remaining,
                "task_tokens_remaining": tb.tokens_remaining,
                "task_exhausted": tb.exhausted,
            })
        return base

    def prioritize(self, task_ids: list[str]) -> list[str]:
        def key(tid):
            tb = self.get_or_create(tid)
            tier = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}.get(task_difficulty(tid), 0)
            return (1 if tb.exhausted else 0, tier)
        return sorted(task_ids, key=key)

    def stats(self) -> dict:
        g = self.gbudget
        return {
            "total_tasks": len(self._task_budgets),
            "completed": sum(1 for t in self._task_budgets.values() if t.exhausted),
            "total_attempts": sum(t.attempts for t in self._task_budgets.values()),
            "csim_used": g.csim_used, "synth_used": g.synth_used,
            "tokens_used": g.tokens_used,
            "csim_remaining": g.csim_remaining, "synth_remaining": g.synth_remaining,
            "tokens_remaining": g.tokens_remaining,
        }