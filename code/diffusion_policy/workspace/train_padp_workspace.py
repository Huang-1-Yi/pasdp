if __name__ == "__main__":
    import sys
    import os
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import math
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import wandb
from termcolor import cprint
import tqdm
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.rollout_early_stopping import (
    update_rollout_early_stopping,
)
from diffusion_policy.workspace.common import BaseTrainingWorkspace, run_rollout

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainPADPWorkspace(BaseTrainingWorkspace):
    include_keys = [
        'global_step', 'epoch', 'lr_step', 'rollout_success_streak'
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.lr_step = 0        # 策略新增，区分预热和非预热阶段的步数

        # 新增：加载预训练模型（其他配置都不变）
        if cfg.training.get('pretrained_model_path', None) is not None:
            pretrained_path = pathlib.Path(cfg.training.pretrained_model_path)
            if pretrained_path.is_file():
                print()
                print(f"正在从 pretrained model 加载权重: {pretrained_path}")
                ckpt = torch.load(pretrained_path, map_location='cpu')
                # 支持两种格式：完整 workspace 格式 或 纯 state_dict 格式
                if 'state_dicts' in ckpt and 'model' in ckpt['state_dicts']:
                    self.model.load_state_dict(ckpt['state_dicts']['model'])
                    if self.ema_model is not None and 'ema_model' in ckpt['state_dicts']:
                        self.ema_model.load_state_dict(ckpt['state_dicts']['ema_model'])
                else:
                    self.model.load_state_dict(ckpt)
                print("Pretrained model 加载成功。")
                print()
            else:
                print(f"Pretrained model 路径不存在: {pretrained_path}，将从零开始训练。")

    def _rollout_enabled(self, cfg: OmegaConf) -> bool:
        """Standard PADP preserves its historical always-on runner."""
        return True

    def _on_epoch_start(self, cfg: OmegaConf) -> None:
        """Variant hook; standard PADP has no epoch-start mutation."""

    @staticmethod
    def _resolve_epoch_batch_plan(dataset, train_dataloader, cfg):
        """Return full and effective batch counts for every training epoch."""
        num_epochs = int(cfg.training.num_epochs)
        get_counts = getattr(dataset, 'get_epoch_batch_counts', None)
        if callable(get_counts):
            full_counts = [
                int(value) for value in get_counts(num_epochs)
            ]
        else:
            full_counts = [int(len(train_dataloader))] * num_epochs

        if len(full_counts) != num_epochs or any(
                value <= 0 for value in full_counts):
            raise RuntimeError(
                'Invalid per-epoch training batch plan: '
                f'expected {num_epochs} positive entries, got {full_counts}'
            )

        max_train_steps = cfg.training.get('max_train_steps', None)
        if max_train_steps is None:
            effective_counts = list(full_counts)
        else:
            cap = int(max_train_steps)
            if cap <= 0:
                raise ValueError('training.max_train_steps must be positive')
            effective_counts = [min(value, cap) for value in full_counts]
        return full_counts, effective_counts

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ######## Debug 配置与 checkpoint 恢复 ########
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print()
                print(f"正在从 checkpoint 恢复训练: {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                print()

        # ######## Dataset、DataLoader 与 normalizer ########
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        # 【v4改动·A.3】worker_init_fn 在 worker 启动时只调一次 threadpool_limits(1)，
        # 支持所有带有 worker_init_fn 静态方法的 dataset（RTV8 Robomimic / MetaWorld / Adroit）。
        _wif = getattr(dataset, 'worker_init_fn', None) \
            if callable(getattr(dataset, 'worker_init_fn', None)) \
            else getattr(type(dataset), 'worker_init_fn', None) \
            if callable(getattr(type(dataset), 'worker_init_fn', None)) \
            else None
        train_dataloader = DataLoader(dataset, worker_init_fn=_wif, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        enable_validation = bool(
            getattr(cfg.training, "enable_validation", True))
        val_dataloader = None
        if enable_validation:
            val_dataset = dataset.get_validation_dataset()
            _wif_val = getattr(val_dataset, 'worker_init_fn', None) \
                if callable(getattr(val_dataset, 'worker_init_fn', None)) \
                else getattr(type(val_dataset), 'worker_init_fn', None) \
                if callable(getattr(type(val_dataset), 'worker_init_fn', None)) \
                else None
            val_dataloader = DataLoader(
                val_dataset, worker_init_fn=_wif_val, **cfg.val_dataloader)

        self._set_normalizer(normalizer)

        # ######## RTV8 训练批次计划与 LR scheduler ########
        # RTV8 映射在生成时已将每个 epoch 的 max_cols 写入
        # manifest。使用全部 epoch 的真实 batch 数规划 cosine
        # scheduler，不截断映射，也不用 epoch 0 长度估算全程。
        epoch_batch_counts, effective_epoch_batch_counts = \
            self._resolve_epoch_batch_plan(dataset, train_dataloader, cfg)
        total_training_batches = int(sum(effective_epoch_batch_counts))
        accumulation_steps = int(cfg.training.gradient_accumulate_every)
        if accumulation_steps <= 0:
            raise ValueError('gradient_accumulate_every must be positive')
        total_optimizer_steps = int(math.ceil(
            total_training_batches / accumulation_steps))
        print(
            'RTV8 training plan: '
            f'epochs={len(epoch_batch_counts)}, '
            f'batches={total_training_batches}, '
            f'optimizer_steps={total_optimizer_steps}, '
            f'epoch_batch_range='
            f'[{min(epoch_batch_counts)}, {max(epoch_batch_counts)}]'
        )

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_optimizer_steps,
            # 使用 batch 级 lr_step 对齐 last_epoch
            last_epoch=self.lr_step-1    # last_epoch由self.global_step-1改为self.lr_step-1
        )

        # ######## EMA、rollout runner 与 WandB ########
        ema = self._create_ema(cfg)

        env_runner = self._create_env_runner(
            cfg, enabled=self._rollout_enabled(cfg)
        )
        if env_runner is None:
            cprint(
                "[workspace] rollout disabled "
                "(training.enable_rollout=false)",
                "yellow",
            )

        # 引入了 termcolor 库来在控制台中输出带颜色的日志信息，使得日志更加易于阅读
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        print("开始设置wandb，并Logging to wandb")
        # 创建 logging 配置的副本
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **logging_config,  # 使用修改后的配置
        )
        wandb.config.update({
                "output_dir": self.output_dir,
                "rtv8_epoch_batch_counts": epoch_batch_counts,
                "rtv8_total_training_batches": total_training_batches,
                "rtv8_total_optimizer_steps": total_optimizer_steps,
        })

        # ######## Checkpoint 管理、训练设备与日志 ########
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = self._move_training_state_to_device(cfg)

        train_sampling_batch = None

        # 创建独立的日志文件路径
        train_log_path = os.path.join(self.output_dir, 'train_logs.json.txt')
        val_log_path = os.path.join(self.output_dir, 'val_logs.json.txt')
        test_log_path = os.path.join(self.output_dir, 'test_logs.json.txt')
        # ######## 逐 epoch 训练、验证、rollout 与保存 ########
        with JsonLogger(train_log_path) as train_logger, \
             JsonLogger(val_log_path) as val_logger, \
            JsonLogger(test_log_path) as test_logger:
            while self.epoch < cfg.training.num_epochs:
                self._on_epoch_start(cfg)

                # ######## 激活当前 epoch 的 RTV8 映射 ########
                epoch_index = int(self.epoch)
                set_epoch = getattr(dataset, 'set_epoch', None)
                if callable(set_epoch):
                    set_epoch(epoch_index)
                epoch_batches = int(len(train_dataloader))
                expected_epoch_batches = int(
                    epoch_batch_counts[epoch_index])
                if epoch_batches != expected_epoch_batches:
                    raise RuntimeError(
                        f'RTV8 epoch {epoch_index}: expected '
                        f'{expected_epoch_batches} batches, got '
                        f'{epoch_batches}'
                    )
                step_log = dict()
                should_early_stop = False
                # ========= train for this epoch ==========
                if getattr(cfg.training, 'freeze_encoder', False):
                    self.model.freeze_obs_encoder()

                # 在 GPU 上累加训练 loss，避免每个 batch 调用 batch_loss.item()
                # 触发 CUDA 同步；epoch 结束后只同步一次，数学均值不变。
                train_loss_sum = torch.zeros((), device=device)
                train_loss_count = 0

                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # ######## 提取 CPU 元数据并将训练张量搬到 GPU ########
                        if 'window_info' in batch:
                            window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                            warm_up_flags = window_info[:, 0, 0]  # [B]
                        
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # 【v4改动·B Step4】uint8_to_gpu 开关打开时，rgb obs 在 GPU 上做归一化：
                        #   uint8 [0,255] -> float32 [0,1]。low_dim/action 不动。
                        #   与 Dataset 关闭开关时 CPU 路径数学等价（uint8 在 GPU /255 = CPU /255 后上传，
                        #   浮点舍入可能有 ulp 级差异）。
                        #   诊断打印在归一化前触发，因此开关 True 时能看到 uint8/255 形态，便于核对。
                        if cfg.training.get('uint8_to_gpu', False):
                            for _k in list(batch['obs'].keys()):
                                _v = batch['obs'][_k]
                                if _v.dtype == torch.uint8:
                                    batch['obs'][_k] = _v.float().div_(255.0)

                        # train_sampling_batch 用于后续 sample 探针；必须在归一化后赋值，
                        # 否则开关 True 时它会存 uint8，后续 sample 阶段直接喂 normalizer 会数值错乱。
                        if train_sampling_batch is None:
                            train_sampling_batch = batch    # 用于采样监控训练进度

                        # 部分2：计算损失并反向传播；每 cfg.training.gradient_accumulate_every 步更新一次优化器和 lr_scheduler
                        # if batch_idx == getattr(self, '_target_sample_idx', -1) or train_sampling_batch is None:
                        #     # detach() 防止某些情况下计算图残留导致显存泄漏
                        #     train_sampling_batch = dict_apply(batch, lambda x: x.detach().clone())
                        
                        # 计算每个样本的损失
                        raw_loss = self.model.compute_loss(batch) # 计算每个样本的损失，返回 [B] 张量

                        # 计算整个批次的平均损失
                        batch_loss = raw_loss.mean()  # 标量
                        
                        loss = batch_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.lr_step += 1
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        # detach 只切断日志累加器的计算图，不改变前向、反向和优化器更新。
                        train_loss_sum += batch_loss.detach()
                        train_loss_count += 1
                        # 这里的 step_log 只含 Python 标量元信息，不读取 GPU tensor，
                        # 所以 wandb 逐步记录不会触发 GPU-CPU 同步。
                        step_log = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # 【v4改动·A.5】原每 step 都 wandb_run.log，改为每 100 step log 一次。
                            # step_log 仅含 global_step/epoch/lr 三个 Python 标量，不读 GPU tensor，
                            # 不触发 .item() 同步；epoch 末的合并日志仍走末尾分支，
                            # 不受节流影响——train_loss / rollout / val 仍正常上报，checkpoint 判据不变。
                            if self.global_step % 100 == 0:
                                wandb_run.log(step_log, step=self.global_step)
                            step_log = {
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                        self.global_step += 1  # ✅ 移动到外层

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # 唯一一次 GPU-CPU 同步：得到本 epoch 的平均训练 loss 用于日志和 checkpoint。
                mean_loss = (train_loss_sum / train_loss_count).item()
                step_log.update({
                    'rtv8/epoch_mapping_batches': epoch_batches,
                    'rtv8/epoch_training_batches':
                        effective_epoch_batch_counts[epoch_index],
                    'rtv8/total_training_batches': total_training_batches,
                    'rtv8/total_optimizer_steps': total_optimizer_steps,
                })
                step_log['train_loss'] = mean_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()


                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    print("Running rollout...")
                    if env_runner is not None:
                        if hasattr(env_runner, 'set_epoch'):
                            env_runner.set_epoch(self.epoch)
                        runner_log = run_rollout(env_runner, policy)
                    else:
                        runner_log = {'test/mean_score': -1.0}
                    
                    # 新增：提取测试分数并记录到测试日志
                    test_score = runner_log.get('test/mean_score', None)  # 注意键名是 'test/mean_score'
                    if test_score is not None:
                        test_log_entry = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'test_mean_score': test_score
                        }
                        for runner_key, log_key in (
                            ('test/success_rate', 'test_success_rate'),
                            ('test/avg_completion_step', 'test_avg_completion_step'),
                            ('test/median_completion_step', 'test_median_completion_step'),
                        ):
                            if runner_key in runner_log:
                                test_log_entry[log_key] = runner_log[runner_key]
                        test_logger.log(test_log_entry)
                        print(f"记录测试分数: {test_score} (epoch {self.epoch})")

                    # log all
                    step_log.update(runner_log)
                    print("Rollout done.")
                    early_stop_result = update_rollout_early_stopping(
                        runner_log=runner_log,
                        config=cfg.training.get('early_stopping', None),
                        current_streak=self.rollout_success_streak,
                    )
                    if early_stop_result is not None:
                        self.rollout_success_streak = early_stop_result.streak
                        should_early_stop = early_stop_result.should_stop
                        step_log['early_stop/score'] = early_stop_result.score
                        step_log['early_stop/streak'] = early_stop_result.streak
                        step_log['early_stop/triggered'] = should_early_stop
                        print(
                            "Early stopping: "
                            f"{early_stop_result.score_key}="
                            f"{early_stop_result.score:.4f}, "
                            f"streak={early_stop_result.streak}/"
                            f"{int(cfg.training.early_stopping.patience)}"
                        )


                """
                # ==========================================================================
                # [排障探针说明]：Validation (验证集开环评估)
                # - 默认状态：【已注释屏蔽】(追求极致训练速度，释放 100% 算力)
                # - 核心逻辑：当前 Checkpoint 保存已完全绑定仿真 Rollout 成功率 (test_mean_score)，
                #   因此纯数学上的验证集 Loss 已无参与选拔的必要。
                  
                # - 🚨 何时需要取消注释 (开启调试)：
                #   当你跑完实验，发现 Rollout 成功率一直是 0% 时。你需要知道是哪里崩了：
                #   1. 若取消注释后，发现 `val_loss` 很低 / 正常下降 -> 说明模型拟合能力没问题，
                #      成功率为 0 是因为仿真环境的 Covariate Shift (OOD) 或物理引擎配置出错。
                #   2. 若取消注释后，发现 `val_loss` 极高或 NaN -> 说明模型超参崩了，网络根本没有
                #      学会拟合人类动作，不要去查物理引擎的 Bug，先调学习率或网络结构。
                # ==========================================================================

                # run validation（仅统计平均验证损失，不做反归一化作图与逐位置MSE）
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        total_non_warmup_loss = torch.zeros((), device=device)
                        total_non_warmup_samples = 0
                        # 新增：按时间步统计 MSE（对 action 维求均值，再对样本求和）
                        # MSE 累加器保持在 GPU 上，避免每个 validation batch 都 .to('cpu') 阻塞。
                        # 只在整个验证循环结束后统一同步一次用于日志，不改变 MSE 计算公式。
                        val_sqerr_sum_per_t = None
                        val_mse_total_samples = 0
                        mean_mse_per_t = None
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                warm_up_flags = None
                                if 'window_info' in batch:
                                    window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                                    # 提取每个序列的第一个时间步的 warm_up_flag
                                    warm_up_flags = window_info[:, 0, 0]  # [B]

                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                realtime_losses = self.model.compute_loss(batch)  # [B]

                                # 逐时间步 MSE: (pred[:, t, :] - gt[:, t, :])^2
                                # 这里使用当前 eval policy（EMA 或在线模型）进行动作预测
                                try:
                                    # policy.reset()
                                    pred_result = policy.predict_action(batch['obs'])
                                    if 'action_pred' in pred_result:
                                        pred_action = pred_result['action_pred']  # [B,H,A]
                                        gt_action = batch['action']               # [B,H,A]

                                        cur_h = min(pred_action.shape[1], gt_action.shape[1])
                                        if cur_h > 0:
                                            sqerr = (pred_action[:, :cur_h, :] - gt_action[:, :cur_h, :]) ** 2
                                            # [B,H]，对 action 维取均值
                                            mse_per_t = sqerr.mean(dim=-1)
                                            # [H]，对 batch 维求和。保持在 GPU 上累加，避免逐 batch CPU 同步。
                                            mse_per_t_sum = mse_per_t.sum(dim=0).detach().to(dtype=torch.float64)

                                            if val_sqerr_sum_per_t is None:
                                                # 第一次按实际 horizon 初始化，device 与 mse_per_t_sum 保持一致。
                                                val_sqerr_sum_per_t = mse_per_t_sum.clone()
                                            else:
                                                # 防御性处理：若 horizon 不一致，按最小长度对齐
                                                if val_sqerr_sum_per_t.shape[0] != mse_per_t_sum.shape[0]:
                                                    min_h = min(val_sqerr_sum_per_t.shape[0], mse_per_t_sum.shape[0])
                                                    val_sqerr_sum_per_t = val_sqerr_sum_per_t[:min_h]
                                                    mse_per_t_sum = mse_per_t_sum[:min_h]
                                                val_sqerr_sum_per_t += mse_per_t_sum

                                            val_mse_total_samples += int(pred_action.shape[0])
                                except Exception as e:
                                    print(f"Warning: failed to compute per-timestep val MSE at batch {batch_idx}: {e}")
                                
                                # 只记录非预热样本的损失，预热期间的损失会被自动滤除
                                # 计算非预热样本的损失
                                if warm_up_flags is not None:
                                    non_warmup_mask = (warm_up_flags == 0)
                                    non_warmup_losses = realtime_losses[non_warmup_mask]
                                else:
                                    non_warmup_losses = realtime_losses
                                
                                # 累加损失和样本数量
                                if len(non_warmup_losses) > 0:
                                    total_non_warmup_loss += non_warmup_losses.sum().detach()
                                    total_non_warmup_samples += len(non_warmup_losses)

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if total_non_warmup_samples > 0:
                            val_loss = (total_non_warmup_loss / total_non_warmup_samples).item()
                        else:
                            val_loss = 0.0
                            print("Warning: No non-warmup samples found in validation.")
                        
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
                        # 记录逐时间步 MSE 到 WandB（通过 step_log 统一上报）
                        if (val_sqerr_sum_per_t is not None) and (val_mse_total_samples > 0):
                            # 唯一一次 MSE 指标同步：验证结束后再搬到 CPU，供 numpy / wandb 日志使用。
                            mean_mse_per_t = (val_sqerr_sum_per_t / float(val_mse_total_samples)).cpu().numpy()
                            for t_idx, t_mse in enumerate(mean_mse_per_t.tolist()):
                                step_log[f'val_mse_t{t_idx}'] = float(t_mse)

                        # 创建专门的验证日志条目
                        val_log_entry = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'val_loss': val_loss
                        }
                        if mean_mse_per_t is not None:
                            val_log_entry['val_mse_per_t'] = mean_mse_per_t.tolist()
                        val_logger.log(val_log_entry)  # 记录到验证日志
                print("Validation done.")

                
                # ==========================================================================
                # [排障探针说明]：Train Sampling (训练集闭环去噪监控)
                # - 默认状态：【已注释屏蔽】(追求极致训练速度)
                # - 核心逻辑：`train_loss` 监控的是“预测加噪误差”，而此模块是让模型在熟悉的
                #   训练集上执行几十步完整的“闭环去噪生成”，监控最终动作的 action_mse。
                  
                # - 🚨 何时需要取消注释 (开启调试)：
                #   当你发现 `val_loss` 正常下降，但 Rollout 成功率依旧极差时。
                #   开启此探针，让模型在最熟悉的训练集数据上走一遍完整的去噪推理：
                #   1. 若 `train_action_mse_error` 很低 -> 扩散去噪过程本身没问题，纯粹是泛化失败。
                #   2. 若 `train_action_mse_error` 很高 -> 扩散模型的生成过程坏了（比如 PADP 的
                #      噪声调度表错误、滑动窗口预热机制失效等）。
                     
                # ⚠️ 警告：取消本模块注释时，务必同步取消内部 `policy.reset()` 的注释！
                # ==========================================================================


                # 在每个 epoch 结束时进行额外的采样、误差计算
                # 对训练集采样的第一个batch，使用当前 policy 进行动作预测，并与真实动作（gt_action）计算均方误差（MSE），将误差记录到日志（step_log['train_action_mse_error']）。
                # 监控模型在训练集上的动作预测效果。
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        policy.reset() # 🚨 注意：如果模型有 reset 机制，务必在采样前调用，否则可能导致采样状态异常，误差过大。
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                        # torch.cuda.empty_cache()
                    """
                
                # checkpoint save
                if ((self.epoch % cfg.training.checkpoint_every) == 0
                        or should_early_stop):
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                train_logger.log(step_log)  # 记录完整的step_log到训练日志

                self.epoch += 1
                if should_early_stop:
                    print(
                        "Stopping training after "
                        f"{self.rollout_success_streak} consecutive fully "
                        "successful rollouts."
                    )
                    break

        # 参考 CLASS 的正常结束路径，显式关闭 runner 持有的 vector env。
        # 异常与 Ctrl+C 的 try/finally 加固留在后续生命周期专项中处理。
        if env_runner is not None:
            env_runner.close()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainPADPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
