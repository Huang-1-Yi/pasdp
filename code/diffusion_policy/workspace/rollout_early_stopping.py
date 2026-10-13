"""Small, policy-independent early stopping helper for rollout metrics."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Optional, Tuple


DEFAULT_SCORE_KEYS: Tuple[str, ...] = (
    "test_mean_score",
    "test/mean_score",
    "test/success_rate",
)


@dataclass(frozen=True)
class RolloutEarlyStoppingResult:
    score: float
    score_key: str
    streak: int
    should_stop: bool


def update_rollout_early_stopping(
        runner_log: Mapping[str, Any],
        config: Optional[Mapping[str, Any]],
        current_streak: int) -> Optional[RolloutEarlyStoppingResult]:
    """Update a consecutive-success streak from one completed rollout.

    Returns ``None`` when early stopping is disabled.  A missing metric is an
    error when enabled so that an experiment cannot silently run forever under
    a misspelled or incompatible runner contract.
    """
    if config is None or not bool(config.get("enabled", False)):
        return None

    threshold = float(config.get("threshold", 1.0))
    patience = int(config.get("patience", 3))
    if not isfinite(threshold):
        raise ValueError(f"early stopping threshold must be finite, got {threshold}")
    if patience <= 0:
        raise ValueError(f"early stopping patience must be positive, got {patience}")

    preferred_key = str(config.get("monitor_key", "test_mean_score"))
    candidate_keys = (preferred_key,) + tuple(
        key for key in DEFAULT_SCORE_KEYS if key != preferred_key)
    score_key = next((key for key in candidate_keys if key in runner_log), None)
    if score_key is None:
        raise KeyError(
            "early stopping could not find a rollout score; tried "
            f"{candidate_keys}, available keys are {tuple(runner_log.keys())}"
        )

    score = float(runner_log[score_key])
    if not isfinite(score):
        raise ValueError(
            f"early stopping metric {score_key!r} must be finite, got {score}"
        )
    streak = int(current_streak) + 1 if score >= threshold else 0
    return RolloutEarlyStoppingResult(
        score=score,
        score_key=score_key,
        streak=streak,
        should_stop=streak >= patience,
    )


__all__ = [
    "DEFAULT_SCORE_KEYS",
    "RolloutEarlyStoppingResult",
    "update_rollout_early_stopping",
]
