import copy
import random
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.envs.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class BaseTrainingWorkspace(BaseWorkspace):
    """Suite-neutral construction helpers shared by diffusion workspaces.

    This class intentionally does not own the training loop or loss reduction.
    DP and PADP keep those algorithm-specific semantics in their public
    workspaces, while sharing deterministic initialization and runtime setup.
    """

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self._seed_everything(int(cfg.training.seed))
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = (
            copy.deepcopy(self.model) if cfg.training.use_ema else None
        )
        self.optimizer = self._create_optimizer(cfg)
        self.global_step = 0
        self.epoch = 0
        self.rollout_success_streak = 0

    @staticmethod
    def _seed_everything(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _create_optimizer(self, cfg: OmegaConf):
        optimizer_source = str(
            cfg.training.get("optimizer_source", "hydra")
        )
        if optimizer_source == "model":
            if not hasattr(self.model, "optimizer"):
                raise AttributeError(
                    "training.optimizer_source=model requires "
                    "policy.optimizer"
                )
            return self.model.optimizer

        if optimizer_source == "policy_method":
            if not hasattr(self.model, "get_optimizer"):
                raise AttributeError(
                    "training.optimizer_source=policy_method requires "
                    "policy.get_optimizer(...)"
                )
            optimizer_cfg = cfg.get("policy_optimizer", cfg.optimizer)
            optimizer_kwargs = OmegaConf.to_container(
                optimizer_cfg, resolve=True
            )
            optimizer_kwargs.pop("_target_", None)
            return self.model.get_optimizer(**optimizer_kwargs)

        if optimizer_source == "hydra":
            return hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters()
            )

        raise ValueError(
            "training.optimizer_source must be one of "
            "hydra, model, policy_method; got "
            f"{optimizer_source!r}"
        )

    def _set_normalizer(self, normalizer) -> None:
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

    def _create_ema(self, cfg: OmegaConf):
        if not cfg.training.use_ema:
            return None
        return hydra.utils.instantiate(cfg.ema, model=self.ema_model)

    def _create_env_runner(
        self,
        cfg: OmegaConf,
        *,
        enabled: bool = True,
    ) -> Optional[BaseImageRunner]:
        if not enabled:
            return None
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseImageRunner)
        return env_runner

    def _move_training_state_to_device(self, cfg: OmegaConf) -> torch.device:
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        return device
