"""
架构设计
    观测编码器：使用robomimic的CNN（可能含CropRandomizer）提取图像特征，支持替换BatchNorm为GroupNorm。
    扩散模型：ConditionalUnet1D处理时间序列动作，支持通过全局条件（观测特征）或局部条件（观测与动作拼接）注入信息。
    条件机制：通过obs_as_global_cond控制是否将观测特征作为全局条件，影响模型输入维度。

关键组件
    调度器：DDPMScheduler管理扩散过程的时间步。
    掩码生成器：LowdimMaskGenerator处理条件掩码，用于训练时数据修补。
    数据增强：RotRandomizer支持旋转增强，CropRandomizer处理图像裁剪。

训练流程
    损失计算：预测噪声与真实噪声的MSE损失，通过掩码忽略条件部分。
    归一化：LinearNormalizer统一处理观测和动作的归一化。

推理流程
    采样过程：通过conditional_sample逐步去噪生成动作序列，最终提取有效动作步。
"""


from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.common.base_diffusion_policy import (
    BaseDiffusionPolicy,
)
from diffusion_policy.model.diffusion_heads.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.components.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.encoders.factory import build_obs_encoder
from diffusion_policy.model.encoders.rgb.rot_randomizer import RotRandomizer


class DiffusionUnetHybridImagePolicy(BaseDiffusionPolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_encoder: Optional[nn.Module] = None,
            rgb_model: str = "robomimic_raw",
            pointcloud_model: str = "pointnet",
            voxel_model: str = "conv3d",
            rgb_encoder_kwargs: Optional[Dict] = None,
            pointcloud_encoder_kwargs: Optional[Dict] = None,
            voxel_encoder_kwargs: Optional[Dict] = None,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,# 设置扩散步骤的嵌入维度，并传递给模型的初始化
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # parameters passed to step
            **kwargs):
        """
        功能：初始化 DiffusionUnetHybridImagePolicy 类，设置模型的所有超参数，加载配置，初始化各类组件。
        参数：
            shape_meta: 包含动作和观察数据的形状信息。
            noise_scheduler: 用于控制扩散过程的调度器(DDPM调度器)。
            horizon: 轨迹的长度。
            n_action_steps: 每次动作的时间步长。
            n_obs_steps: 观察数据的时间步长。
            其他相关的配置参数，如 crop_shape、diffusion_step_embed_dim 等。
        关键步骤：
            解析形状元数据 (shape_meta)。
            配置观察数据的类型和模态（RGB、低维、深度等)。
            加载 Robomimic 配置，设定观察数据的随机化方式（如裁剪）。
            初始化 obs_encoder（用于处理观察数据）和 model（扩散模型）。
            初始化 noise_scheduler 和 mask_generator 等其他相关组件。
        """
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        if obs_encoder is None:
            encoder_kwargs = dict(rgb_encoder_kwargs or {})
            encoder_kwargs.setdefault("crop_shape", crop_shape)
            encoder_kwargs.setdefault("eval_fixed_crop", eval_fixed_crop)
            if rgb_model == "robomimic_raw":
                encoder_kwargs.setdefault(
                    "obs_encoder_group_norm", obs_encoder_group_norm)
            elif rgb_model.startswith("timm"):
                encoder_kwargs.setdefault(
                    "use_group_norm", obs_encoder_group_norm)
            obs_encoder = build_obs_encoder(
                shape_meta=shape_meta,
                rgb_model=rgb_model,
                pointcloud_model=pointcloud_model,
                voxel_model=voxel_model,
                rgb_encoder_kwargs=encoder_kwargs,
                pointcloud_encoder_kwargs=pointcloud_encoder_kwargs,
                voxel_encoder_kwargs=voxel_encoder_kwargs,
            )

        # create diffusion model
        output_shape = obs_encoder.output_shape()
        if isinstance(output_shape, int):
            obs_feature_dim = int(output_shape)
        elif len(output_shape) == 1:
            obs_feature_dim = int(output_shape[0])
        else:
            raise ValueError(
                "obs_encoder.output_shape() must be one-dimensional, got "
                f"{output_shape}"
            )
        print(
            "Observation encoder selection: "
            f"rgb={rgb_model}, pointcloud={pointcloud_model}, "
            f"voxel={voxel_model}, "
            f"class={type(obs_encoder).__module__}.{type(obs_encoder).__name__}, "
            f"model_name={getattr(obs_encoder, 'model_name', None)}, "
            f"output_shape={output_shape}"
        )
        # 代码11 - 简单的条件输入处理 输入包括动作和观察的联合特征
        # 代码11 总是将动作和观察数据联合处理，因此输入维度为动作维度加上观察特征维度
        input_dim = action_dim + obs_feature_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            # obs_as_global_cond如果为 True，仅使用观察数据的特征作为条件输入；
            # obs_as_global_cond如果为 False，将观察特征和动作特征结合作为输入
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self._initialize_normalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Observation encoder params: %e" % sum(
            p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        功能：基于给定的条件数据和条件掩码，通过扩散模型进行采样。
        参数：
            condition_data、condition_mask: 条件数据和掩码。
            local_cond、global_cond: 本地条件和全局条件（根据 obs_as_global_cond 来决定）。
            generator: 随机数生成器，用于采样。
        关键步骤：
            初始化扩散轨迹。
            设置调度器的时间步。
            在每个时间步上，应用条件数据和掩码进行采样。
            调用扩散模型的输出，并通过调度器一步步反向扩散生成目标数据。
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        功能：根据当前观察数据预测动作。
        参数：
            obs_dict: 包含观察数据的字典，必须包括键 "obs"。
        关键步骤：
            1.对观察数据进行归一化处理。
            2.构建输入数据（观察和动作数据）。
            3.根据是否使用全局条件 (obs_as_global_cond)，分别处理本地条件和全局条件。
            4.调用 conditional_sample 方法进行条件采样，生成轨迹。
            5.生成预测的动作，并将其反归一化（恢复到原始尺度）。
            6.返回预测的动作。
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalize_obs(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            nobs_features = self._encode_normalized_obs(nobs, n_steps=To)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            nobs_features = self._encode_normalized_obs(nobs, n_steps=To)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.unnormalize_action(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def reset(self):
        """DP inference is stateless between rollout calls."""
        pass

    # ========= training  ============

    def compute_loss(self, batch):
        """
        功能：计算模型的损失函数。
        参数：
            batch: 输入的批次数据，包含观察数据和动作数据。
        关键步骤：
            1.对输入数据进行归一化处理。
            2.根据是否使用旋转增强 (rot_aug)，对数据进行旋转增强。
            3.处理观察数据，通过全局条件或其他方式构建条件数据。
            4.使用 mask_generator 生成损失掩码。
            5.计算噪声并添加到图像中，模拟扩散过程。
            6.根据预测类型（epsilon 或 sample）计算目标（噪声或轨迹）。
            7.使用均方误差（MSE）损失函数计算损失。
            8.计算和返回最终的平均损失。
        """
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalize_obs(batch['obs'])
        nactions = self.normalize_action(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            nobs_features = self._encode_normalized_obs(
                nobs, n_steps=self.n_obs_steps)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            nobs_features = self._encode_normalized_obs(nobs, n_steps=None)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


# Environment-neutral public name; retain the historical class above for
# checkpoint and downstream import compatibility.
DPPolicy = DiffusionUnetHybridImagePolicy
