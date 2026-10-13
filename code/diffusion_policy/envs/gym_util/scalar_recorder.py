"""Small metric recorder migrated from FocalPolicy's logger utility."""

import heapq


class LargestKRecorder:
    def __init__(self, k: int):
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = int(k)
        self.scalars = []

    def record(self, scalar: float) -> None:
        scalar = float(scalar)
        if len(self.scalars) < self.k:
            heapq.heappush(self.scalars, scalar)
        elif scalar > self.scalars[0]:
            heapq.heappushpop(self.scalars, scalar)

    def average(self) -> float:
        if not self.scalars:
            raise ValueError("no scalars have been recorded")
        return sum(self.scalars) / len(self.scalars)


__all__ = ["LargestKRecorder"]
