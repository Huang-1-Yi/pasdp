"""Shared environment-rollout call used by training and evaluation."""

from typing import Any, Mapping


def run_rollout(env_runner, policy) -> Mapping[str, Any]:
    """Run the configured policy once without changing runner semantics.

    Resource ownership, epoch propagation, policy selection, logging, early
    stopping, and checkpointing remain the caller's responsibility. Keeping
    this function as a direct forwarding call guarantees that extraction does
    not add resets or other policy/environment computations.
    """

    return env_runner.run(policy)


__all__ = ["run_rollout"]
