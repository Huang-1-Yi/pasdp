import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.envs.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.envs.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.envs.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.envs.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.common.base_policy import BasePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.envs.base_image_runner import BaseImageRunner
from diffusion_policy.envs.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
# 可选引入绘图库（若不存在则仅保存NPY）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
# 显式触发 Coffee_D0、Square_D0、StackThree_D0 等 MimicGen 环境注册。
# 注册失败应在导入时暴露，不能静默把任务替换成语义不同的 robosuite 基础任务。
import mimicgen_envs  # noqa: F401

import time

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            debug_save_actions: bool = False,
            # debug_save_actions: bool = True,
            debug_save_details: bool = False,
            debug_save_states: bool = False,  # 新增：保存机器人状态
            # debug_save_states: bool = True,  # 新增：保存机器人状态
            debug_check_actions: bool = False,
            state_save_dir: str = None,  # 修改默认值为None
            epoch=None,   # <--- 【新增这一行】
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # # 新增：创建状态保存目录
        # if debug_save_states:
        #     os.makedirs(state_save_dir, exist_ok=True)
        #     self.state_save_dir = state_save_dir
        # 修改状态保存目录设置
        if debug_save_states:
            if state_save_dir is None:
                # 自动创建与media目录并列的robot_states目录
                state_save_dir = os.path.join(output_dir, 'robot_states')
            os.makedirs(state_save_dir, exist_ok=True)
            self.state_save_dir = state_save_dir
        self.debug_save_states = debug_save_states

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                # policy 需要 RGB observation，因此真正执行 rollout 的
                # worker 必须启用 offscreen renderer；是否录制视频由
                # n_train_vis / n_test_vis 单独控制。
                enable_render=True,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # dummy env 只用于让 AsyncVectorEnv 读取 observation/action space，
        # 不会 reset 或 step，因此无需 image observation，也不应在主进程
        # 创建 MuJoCo offscreen OpenGL context。真正的 worker 仍由 env_fn
        # 使用 enable_render=True 创建，评估输入不受影响。
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        
        # --- 替换开始：只收集状态和参数，不再写死闭包 ---
        self.env_seeds = list()
        self.env_prefixs = list()
        self.train_init_states = list()

        # 收集 train 数据
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                self.train_init_states.append(f[f'data/demo_{train_idx}/states'][0])
                self.env_seeds.append(train_idx)
                self.env_prefixs.append('train/')
        
        # 收集 test 数据
        for i in range(n_test):
            seed = test_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append('test/')

        # 绑定核心参数到 self，供 set_epoch 使用
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.output_dir = output_dir

        # 初始化时调用一次，生成带有初始前缀的 env_init_fn_dills
        self.set_epoch(epoch)
        env_init_fn_dills = self.env_init_fn_dills
        # --- 替换结束 ---

        # TODO(v5 后续实验)：保持 shared_memory=True，只对 copy=True/False
        # 做独立 A/B；在验证共享视图生命周期前不得与 non_blocking H2D
        # 或异步 env.step 同时修改。
        env = AsyncVectorEnv(
            env_fns,
            dummy_env_fn=dummy_env_fn,
            shared_memory=True,
            copy=True,
        )
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.debug_save_actions = debug_save_actions
        self.debug_save_details = debug_save_details
        self.debug_check_actions = bool(debug_check_actions)
        self._near_zero_action_warned = False

    def set_epoch(self, epoch):
        self.epoch = epoch
        # 如果 epoch 有值，前缀如 "epoch5_"；否则为空
        ep_str = f"epoch{epoch}_" if epoch is not None else ""
        self.env_init_fn_dills = []

        # 动态重建 train 的初始化函数
        for i in range(self.n_train):
            init_state = self.train_init_states[i]
            enable_render = i < self.n_train_vis
            
            # 这里的魔法在于：通过 i=i, ep_str=ep_str, out_dir=self.output_dir 强制锁死变量值
            def init_fn(env, init_state=init_state, enable_render=enable_render, i=i, ep_str=ep_str, out_dir=self.output_dir):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(out_dir).joinpath(
                        'media', f"{ep_str}train_{i}_{wv.util.generate_id()}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = init_state
                
            self.env_init_fn_dills.append(dill.dumps(init_fn))
            
        # 动态重建 test 的初始化函数
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis
            
            def init_fn(env, seed=seed, enable_render=enable_render, i=i, ep_str=ep_str, out_dir=self.output_dir):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(out_dir).joinpath(
                        'media', f"{ep_str}test_{i}_{wv.util.generate_id()}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)
                
            self.env_init_fn_dills.append(dill.dumps(init_fn))

    def close(self):
        """幂等关闭 vector env 及其 worker、pipe 和底层仿真环境。"""
        env = getattr(self, 'env', None)
        if env is None:
            return
        # 先清空引用，确保底层 close 抛错时再次调用也不会重复关闭。
        self.env = None
        env.close()

    @staticmethod
    def _slice_action_horizon(action, n_action_steps):
        """选择交给 MultiStepWrapper 的前 N 步动作，不足时明确失败。"""
        if action.ndim != 3:
            raise ValueError(
                f"Policy action must have shape [B, T, D], got {action.shape}"
            )
        if action.shape[1] < n_action_steps:
            raise ValueError(
                f"Policy returned {action.shape[1]} action steps, "
                f"but runner requires {n_action_steps}; refusing to tile actions."
            )
        return action[:, :n_action_steps, :]

    def _debug_check_action_l2(self, action):
        """仅在 debug_check_actions=True 时执行 near-zero 动作诊断。"""
        if self._near_zero_action_warned:
            return
        l2 = np.linalg.norm(action[0], axis=-1)
        if np.all(l2 < 1e-4):
            print("[warn] Predicted action L2 is near zero across steps; "
                  "policy may be outputting (almost) zeros.")
            self._near_zero_action_warned = True

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        if env is None:
            raise RuntimeError("Runner has been closed and cannot run again.")
        self._near_zero_action_warned = False

        # 初始化时间统计变量
        total_policy_time = 0.0
        policy_calls = 0

        # 一次性检查并提示 policy 与 runner 的时序设置
        pol_n_act = getattr(policy, 'n_action_steps', None)
        pol_n_obs = getattr(policy, 'n_obs_steps', None)
        if (pol_n_act is not None) and (pol_n_act != self.n_action_steps):
            print(f"[warn] policy.n_action_steps={pol_n_act} != "
                  f"runner.n_action_steps={self.n_action_steps}. "
                  "Runner will execute the first required steps; shorter output is rejected.")
        if (pol_n_obs is not None) and (pol_n_obs != self.n_obs_steps):
            print(f"[warn] policy.n_obs_steps={pol_n_obs} != runner.n_obs_steps={self.n_obs_steps}. Ensure config alignment to avoid conditioning mismatch.")
        print(f"[info] abs_action={self.abs_action}, max_steps={self.max_steps}, n_envs={len(self.env_fns)}") 

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_success_metrics = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            # 新增：初始化状态记录列表
            if self.debug_save_states:
                chunk_states = [[] for _ in range(this_n_active_envs)]  # 每个环境一个状态列表
                chunk_timesteps = [[] for _ in range(this_n_active_envs)]  # 时间步记录
            
            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False

            # 调试：按chunk收集每步准备送入环境的动作
            if self.debug_save_actions:
                print("[debug] rollout action recording is enabled; artifacts will be saved under 'rollout_debug'.")
                chunk_actions = []  # list of np.ndarray [n_envs, n_action_steps, act_dim]
                l2_curve = []      # list of scalar（跨env和步长平均的范数）
            if self.debug_save_details:
                # 更细的 per-env 指标（只记录每轮的第1子步，便于定位）
                firststep_l2_model = []      # [T, n_envs]
                firststep_l2_env = []        # [T, n_envs]
                firststep_l2_model_pos = []  # [T, n_envs]
                firststep_l2_env_pos = []    # [T, n_envs]
                firststep_l2_model_rot = []  # [T, n_envs]
                firststep_l2_env_rot = []    # [T, n_envs]
                firststep_l2_model_grp = []  # [T, n_envs]
                firststep_l2_env_grp = []    # [T, n_envs]
                obs_mean_per_env = []        # [T, n_envs]
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # 测量策略执行时间
                start_time = time.perf_counter()  # 高精度计时器
                # run policy
                with torch.inference_mode():
                    action_dict = policy.predict_action(obs_dict)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                # 累加时间和调用次数
                total_policy_time += elapsed
                policy_calls += 1

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                # 明确选取交给 MultiStepWrapper 的执行窗口；动作不足时禁止
                # np.tile 伪造重复动作，以免掩盖 policy / runner 配置错误。
                action = self._slice_action_horizon(action, self.n_action_steps)

                if self.debug_save_details:
                    try:
                        # 模型输出（对齐步长后）的第1子步 L2（每环境）及分量
                        l2_model = np.linalg.norm(action[:, 0, :], axis=-1)
                        D = action.shape[-1]
                        # 约定：pos(0:3), rot(3:D-1) if D>4, gripper(-1)
                        pos_model = np.linalg.norm(action[:, 0, :3], axis=-1)
                        grp_model = np.linalg.norm(action[:, 0, -1:], axis=-1)
                        if D > 4:
                            rot_model = np.linalg.norm(action[:, 0, 3:-1], axis=-1)
                        else:
                            rot_model = np.zeros_like(pos_model)
                        firststep_l2_model.append(l2_model.copy())
                        firststep_l2_model_pos.append(pos_model.copy())
                        firststep_l2_model_rot.append(rot_model.copy())
                        firststep_l2_model_grp.append(grp_model.copy())
                    except Exception:
                        pass

                # near-zero L2 是排障逻辑，默认关闭；False 路径只有一次布尔判断。
                if self.debug_check_actions:
                    self._debug_check_action_l2(action)
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                # 新增：在执行环境步骤前记录当前状态
                # if self.debug_save_states:
                #     try:
                #         # 获取当前所有环境的状态
                #         states = env.call('get_attr', 'state')
                #         for i in range(this_n_active_envs):
                #             if i < len(states):
                #                 # 记录状态和时间步
                #                 chunk_states[i].append(states[i].copy())
                #                 chunk_timesteps[i].append(pbar.n)
                #     except Exception as e:
                #         print(f"[warn] Failed to record state data: {e}")
                # 新增：在执行环境步骤前记录当前状态
                if self.debug_save_states:
                    try:
                        # 获取当前所有环境的状态
                        states = env.call_each('get_robot_state')
                        for i in range(this_n_active_envs):
                            if i < len(states) and states[i] is not None:
                                # 记录状态和时间步
                                chunk_states[i].append(states[i].copy())
                                chunk_timesteps[i].append(pbar.n)
                    except Exception as e:
                        print(f"[warn] Failed to record state data: {e}")
                                
                # 调试：保存准备送入环境的动作（记录实际将送入环境的 env_action）
                if self.debug_save_actions:
                    try:
                        chunk_actions.append(env_action.copy())
                        # 计算L2范数（先对最后一维求范数-> [n_envs, n_action_steps]，再整体平均到标量）
                        l2 = np.linalg.norm(env_action, axis=-1).mean()
                        l2_curve.append(float(l2))
                    except Exception as e:
                        print(f"[warn] failed to record env actions: {e}")

                if self.debug_save_details:
                    try:
                        # env_action 的第1子步 L2（每环境）及分量
                        l2_env = np.linalg.norm(env_action[:, 0, :], axis=-1)
                        D = env_action.shape[-1]
                        pos_env = np.linalg.norm(env_action[:, 0, :3], axis=-1)
                        grp_env = np.linalg.norm(env_action[:, 0, -1:], axis=-1)
                        if D > 4:
                            rot_env = np.linalg.norm(env_action[:, 0, 3:-1], axis=-1)
                        else:
                            rot_env = np.zeros_like(pos_env)
                        firststep_l2_env.append(l2_env.copy())
                        firststep_l2_env_pos.append(pos_env.copy())
                        firststep_l2_env_rot.append(rot_env.copy())
                        firststep_l2_env_grp.append(grp_env.copy())
                    except Exception:
                        pass
                    try:
                        key = self.env_meta.get('render_obs_key', None) or 'agentview_image'
                        if key in np_obs_dict:
                            img = np_obs_dict[key]
                            if img.ndim >= 3:
                                last = img[:, -1]
                                obs_mean = last.reshape(last.shape[0], -1).mean(axis=1)
                                obs_mean_per_env.append(obs_mean.copy())
                    except Exception:
                        pass

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # success 在各 worker 内按 primitive step 累计。每个 chunk
            # 结束后只做一次 IPC，取回 active env 的最终结果。
            chunk_success_metrics = env.call("get_success_metrics")
            all_success_metrics[this_global_slice] = \
                chunk_success_metrics[this_local_slice]

            # 新增：保存状态数据
            if self.debug_save_states:
                try:
                    for i in range(this_n_active_envs):
                        if chunk_states[i]:  # 如果有记录数据
                            # 转换为numpy数组
                            states_arr = np.array(chunk_states[i])
                            timesteps_arr = np.array(chunk_timesteps[i])
                            
                            # 保存状态数据
                            state_file = os.path.join(
                                self.state_save_dir, 
                                f'chunk_{chunk_idx+1:02d}_env_{i:02d}_states.npy'
                            )
                            np.save(state_file, states_arr)
                            
                            # 保存时间步数据
                            timestep_file = os.path.join(
                                self.state_save_dir,
                                f'chunk_{chunk_idx+1:02d}_env_{i:02d}_timesteps.npy'
                            )
                            np.save(timestep_file, timesteps_arr)
                            
                    print(f"[info] Saved state data for chunk {chunk_idx+1}")
                except Exception as e:
                    print(f"[error] Failed to save state data: {e}")

            # 调试：按chunk落盘动作序列与L2曲线图
            if self.debug_save_actions:
                try:
                    debug_dir = pathlib.Path(self.output_dir).joinpath('rollout_debug')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    # 保存动作序列
                    # 形状: [T_chunk, n_envs, n_action_steps, act_dim]
                    if len(chunk_actions) > 0:
                        actions_arr = np.stack(chunk_actions, axis=0)
                        np.save(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_actions.npy')), actions_arr)
                        # 保存L2曲线图
                        if plt is not None:
                            plt.figure()
                            plt.plot(l2_curve, label='mean L2 over envs/steps')
                            plt.title(f"chunk {chunk_idx+1}/{n_chunks}")
                            plt.xlabel('control loop step')
                            plt.ylabel('L2(action)')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.savefig(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_action_l2.png')), dpi=150, bbox_inches='tight')
                            plt.close()
                except Exception as e:
                    print(f"[warn] failed to save rollout debug artifacts: {e}")

            if self.debug_save_details:
                try:
                    debug_dir = pathlib.Path(self.output_dir).joinpath('rollout_debug')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    def _maybe_save(name, lst):
                        if len(lst) > 0:
                            arr = np.stack(lst, axis=0)
                            np.save(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_{name}.npy')), arr)
                    _maybe_save('firststep_l2_model', firststep_l2_model)
                    _maybe_save('firststep_l2_env', firststep_l2_env)
                    _maybe_save('firststep_l2_model_pos', firststep_l2_model_pos)
                    _maybe_save('firststep_l2_env_pos', firststep_l2_env_pos)
                    _maybe_save('firststep_l2_model_rot', firststep_l2_model_rot)
                    _maybe_save('firststep_l2_env_rot', firststep_l2_env_rot)
                    _maybe_save('firststep_l2_model_grp', firststep_l2_model_grp)
                    _maybe_save('firststep_l2_env_grp', firststep_l2_env_grp)
                    _maybe_save('obs_mean_agentview_image', obs_mean_per_env)
                except Exception as e:
                    print(f"[warn] failed to save per-env debug metrics: {e}")

            # collect data for this round，但是无法收集完整的奖励
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        
        # log
        max_rewards = collections.defaultdict(list)
        test_successes = []
        test_completion_steps = []
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            success_metrics = all_success_metrics[i]
            if prefix == 'test/':
                success = bool(success_metrics["success"])
                first_success_step = success_metrics["first_success_step"]
                completion_step = (
                    int(first_success_step)
                    if success and (first_success_step is not None)
                    else int(self.max_steps)
                )
                test_successes.append(float(success))
                test_completion_steps.append(completion_step)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        if len(test_successes) > 0:
            log_data['test/success_rate'] = float(np.mean(test_successes))
            log_data['test/avg_completion_step'] = float(
                np.mean(test_completion_steps)
            )
            log_data['test/median_completion_step'] = float(
                np.median(test_completion_steps)
            )

        # 添加策略执行时间指标
        if policy_calls > 0:
            avg_policy_time = total_policy_time / policy_calls
            log_data['policy/avg_inference_time_ms'] = avg_policy_time * 1000
            
        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    
