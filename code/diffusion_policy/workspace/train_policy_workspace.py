if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
from typing import Any, Dict, Tuple

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
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.common import BaseTrainingWorkspace, run_rollout
from diffusion_policy.workspace.rollout_early_stopping import (
    update_rollout_early_stopping,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainPolicyWorkspace(BaseTrainingWorkspace):
    include_keys = ["global_step", "epoch", "rollout_success_streak"]

    @staticmethod
    def _unpack_loss(loss_result: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
            loss_info = loss_result[1] if len(loss_result) > 1 else {}
            if not isinstance(loss_info, dict):
                loss_info = {}
        else:
            loss = loss_result
            loss_info = {}
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        return loss, loss_info

    @staticmethod
    def _to_log_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().item()
            return value.detach().cpu().tolist()
        return value


    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        # Optional: balanced-sampler variants provide get_train_dataloader()
        # which builds the DataLoader with batch_sampler= instead of shuffle=True.
        if hasattr(dataset, 'get_train_dataloader'):
            print('[Workspace] Using dataset.get_train_dataloader() (custom sampler)')
            train_dataloader = dataset.get_train_dataloader(**cfg.dataloader)
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        enable_validation = bool(
            getattr(cfg.training, "enable_validation", True))
        val_dataloader = None
        if enable_validation:
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self._set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            )
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema = self._create_ema(cfg)

        env_runner = self._create_env_runner(cfg)

        cfg.logging.name = str(cfg.logging.name)
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **logging_config,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = self._move_training_state_to_device(cfg)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        freeze_encoder = bool(getattr(cfg.training, "freeze_encoder", False))

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = {}
                should_early_stop = False

                if freeze_encoder:
                    self.model.freeze_obs_encoder()

                # Per-epoch reset of any balanced batch sampler (no-op for
                # the default RandomSampler path).  Keeps a ref alive to
                # avoid GC of the sampler between epochs.
                bs = getattr(train_dataloader, 'batch_sampler', None)
                if bs is not None and hasattr(bs, 'set_epoch'):
                    self._train_batch_sampler = bs
                    self._train_batch_sampler.set_epoch(self.epoch)

                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss_result = self.model.compute_loss(batch)
                        raw_loss, loss_info = self._unpack_loss(raw_loss_result)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.detach().item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        for key, value in loss_info.items():
                            step_log[key] = self._to_log_value(value)

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)

                        self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if self.epoch % cfg.training.rollout_every == 0:
                    print("Running rollout...")
                    runner_log = run_rollout(env_runner, policy)
                    step_log.update(runner_log)
                    print("Rollout done.")
                    early_stop_result = update_rollout_early_stopping(
                        runner_log=runner_log,
                        config=cfg.training.get("early_stopping", None),
                        current_streak=self.rollout_success_streak,
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
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )
                                loss_result = self.model.compute_loss(batch)
                                loss, _ = self._unpack_loss(loss_result)
                                val_losses.append(loss.detach())
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= (cfg.training.max_val_steps - 1)
                                ):
                                    break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = torch.stack(val_losses).mean().item()

                if self.epoch % cfg.training.sample_every == 0:
                    with torch.no_grad():
                        # Stateful streaming policies keep an action buffer.
                        # Training-sample diagnostics must start a fresh episode.
                        policy.reset()
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]
                        result = policy.predict_action(obs_dict)
                        pred_action = result.get("action_pred", result["action"])
                        # Chunking policies such as ACT may expose only the
                        # executable action slice. Compare the common prefix.
                        common_steps = min(
                            pred_action.shape[1], gt_action.shape[1])
                        mse = torch.nn.functional.mse_loss(
                            pred_action[:, :common_steps],
                            gt_action[:, :common_steps],
                        )
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if (self.epoch % cfg.training.checkpoint_every == 0
                        or should_early_stop):
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {}
                    for key, value in step_log.items():
                        metric_dict[key.replace("/", "_")] = value
                    if "val_loss" not in metric_dict:
                        metric_dict["val_loss"] = 0.0

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
                if should_early_stop:
                    print(
                        "Stopping training after "
                        f"{self.rollout_success_streak} consecutive fully "
                        "successful rollouts."
                    )
                    break

        # 正常训练结束后，显式关闭 runner 持有的 vector env。
        env_runner.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainPolicyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
