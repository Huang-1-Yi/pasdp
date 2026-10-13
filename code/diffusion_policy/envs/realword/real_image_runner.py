"""Offline-training placeholder, corresponding to 0518's real image runner."""

from diffusion_policy.envs.base_image_runner import BaseImageRunner


class RealImageRunner(BaseImageRunner):
    def run(self, policy):
        # Real closed-loop evaluation is performed separately at deployment.
        return {}
