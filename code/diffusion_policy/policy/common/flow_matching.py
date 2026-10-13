"""Small consistency-flow schedule used by FlowPolicy inference."""


class ConsistencyFlowSchedule:
    """Hold the scalar schedule used by the upstream FlowPolicy sampler."""

    def __init__(
            self,
            noise_scale: float = 1.0,
            sigma_variance: float = 0.0,
            sample_steps: int = 1):
        sample_steps = int(sample_steps)
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        self.noise_scale = float(noise_scale)
        self.sigma_variance = float(sigma_variance)
        self.sample_steps = sample_steps

    def sigma(self, time):
        """Return upstream ``(1 - t) * sigma_variance`` noise scale."""
        return (1.0 - time) * self.sigma_variance


__all__ = ["ConsistencyFlowSchedule"]
