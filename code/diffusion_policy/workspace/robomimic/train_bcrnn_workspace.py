"""Training workspace for native Robomimic image policies such as BCRNN."""

import copy
import os
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.envs.base_image_runner import BaseImageRunner
from diffusion_policy.policy.robomimic_policy import RobomimicImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.rollout_early_stopping import (
    update_rollout_early_stopping,
)
from diffusion_policy.workspace.common import run_rollout


class TrainBCRNNWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "rollout_success_streak"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: RobomimicImagePolicy = hydra.utils.instantiate(cfg.policy)
        # Robomimic owns optimization internally.  Keeping this reference lets
        # BaseWorkspace include optimizer state in checkpoints.
        self.optimizer = self.model.get_optimizer()
        self.global_step = 0
        self.epoch = 0
        self.rollout_success_streak = 0

    @staticmethod
    def _action_loss(info: dict) -> torch.Tensor:
        try:
            return info["losses"]["action_loss"]
        except KeyError as exc:
            raise KeyError(
                "Robomimic training output has no losses.action_loss; "
                f"top-level keys are {tuple(info.keys())}"
            ) from exc

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError(f"dataset must be BaseImageDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        enable_validation = bool(
            getattr(cfg.training, "enable_validation", True))
        val_dataloader = None
        if enable_validation:
            val_dataloader = DataLoader(
                dataset.get_validation_dataset(), **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir)
        if not isinstance(env_runner, BaseImageRunner):
            raise TypeError(
                f"env_runner must be BaseImageRunner, got {type(env_runner)}")

        cfg.logging.name = str(cfg.logging.name)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **OmegaConf.to_container(cfg.logging, resolve=True),
        )
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        train_sampling_batch = None

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = {}
                should_early_stop = False
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training BCRNN epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        info = self.model.train_on_batch(batch, epoch=self.epoch)
                        loss_value = float(self._action_loss(info).detach().item())
                        train_losses.append(loss_value)
                        step_log = {
                            "train_loss": loss_value,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }
                        tepoch.set_postfix(loss=loss_value, refresh=False)
                        is_last_batch = batch_idx == len(train_dataloader) - 1
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                        self.global_step += 1
                        if (cfg.training.max_train_steps is not None
                                and batch_idx >= cfg.training.max_train_steps - 1):
                            break

                step_log["train_loss"] = float(np.mean(train_losses))
                self.model.eval()

                if self.epoch % cfg.training.rollout_every == 0:
                    print("Running BCRNN rollout...")
                    runner_log = run_rollout(env_runner, self.model)
                    step_log.update(runner_log)
                    early_stop_result = update_rollout_early_stopping(
                        runner_log,
                        cfg.training.get("early_stopping", None),
                        self.rollout_success_streak,
                    )
                    if early_stop_result is not None:
                        self.rollout_success_streak = early_stop_result.streak
                        should_early_stop = early_stop_result.should_stop
                        step_log["early_stop/score"] = early_stop_result.score
                        step_log["early_stop/streak"] = early_stop_result.streak
                        step_log["early_stop/triggered"] = should_early_stop
                        print(
                            "Early stopping: "
                            f"{early_stop_result.score_key}="
                            f"{early_stop_result.score:.4f}, "
                            f"streak={early_stop_result.streak}/"
                            f"{int(cfg.training.early_stopping.patience)}"
                        )

                if (enable_validation
                        and self.epoch % cfg.training.val_every == 0):
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True))
                            info = self.model.train_on_batch(
                                batch, epoch=self.epoch, validate=True)
                            val_losses.append(
                                self._action_loss(info).detach().float().cpu())
                            if (cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1):
                                break
                    if val_losses:
                        step_log["val_loss"] = torch.stack(val_losses).mean().item()

                if (train_sampling_batch is not None
                        and self.epoch % cfg.training.sample_every == 0):
                    with torch.no_grad():
                        obs_dict = train_sampling_batch["obs"]
                        gt_action = train_sampling_batch["action"]
                        self.model.reset()
                        predictions = []
                        for time_idx in range(gt_action.shape[1]):
                            one_obs = dict_apply(
                                obs_dict, lambda x: x[:, [time_idx]])
                            predictions.append(
                                self.model.predict_action(one_obs)["action"])
                        action_pred = torch.cat(predictions, dim=1)
                        step_log["train_action_mse_error"] = (
                            torch.nn.functional.mse_loss(
                                action_pred, gt_action).item()
                        )

                # Robomimic advances its internally owned LR schedulers here;
                # do this before checkpointing so resume restores the state at
                # the actual end of the epoch.
                self.model.on_epoch_end(self.epoch)

                checkpoint_due = (
                    self.epoch % cfg.training.checkpoint_every == 0
                    or should_early_stop
                )
                if checkpoint_due:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in step_log.items()
                    }
                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)

                self.model.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
                if should_early_stop:
                    print(
                        "Stopping BCRNN training after "
                        f"{self.rollout_success_streak} consecutive fully "
                        "successful rollouts."
                    )
                    break

        env_runner.close()


__all__ = ["TrainBCRNNWorkspace"]
