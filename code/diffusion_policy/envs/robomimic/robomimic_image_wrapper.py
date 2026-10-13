# 只生成单个相机的可视化图像
from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        # eqdp 单相机
        self.render_obs_key = render_obs_key
        # # ds
        # self.render_obs_keys = [render_obs_key] if isinstance(render_obs_key, str) else render_obs_key

        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        # Episode success 由 worker 内部逐 primitive step 累计，避免主进程
        # 为成功判据额外进行逐步 IPC。失败 episode 的 max_steps 映射由
        # runner 在聚合阶段完成。
        self.episode_step = 0
        self.episode_success = False
        self.first_success_step = None
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            # eqdp
            elif key.endswith('depth'):
                min_value, max_value = 0, 1
            elif key.endswith('voxels'):
                min_value, max_value = 0, 1
            elif key.endswith('point_cloud'):
                min_value, max_value = -10, 10
            # eqdp 截至

            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        # # ds
        # imgs = [raw_obs[key] for key in self.render_obs_keys]
        # self.render_cache = np.concatenate(imgs, axis=1)

        # eqdp设置后仅支持单摄像头
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        self.episode_step = 0
        self.episode_success = False
        self.first_success_step = None

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.episode_step += 1
        if not self.episode_success:
            success_metrics = self.env.is_success()
            task_success = bool(success_metrics["task"])
            if task_success:
                self.episode_success = True
                self.first_success_step = self.episode_step
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def get_success_metrics(self):
        """返回当前 episode 的累计成功状态和首次成功 primitive step。"""
        return {
            "success": self.episode_success,
            "first_success_step": self.first_success_step,
            "episode_step": self.episode_step,
        }
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img

    def get_robot_state(self):
        """
        获取完整的机器人状态，包括位置和姿态
        返回: [x, y, z, qw, qx, qy, qz, gripper_open] 格式
        """
        try:
            # 尝试从底层环境获取完整状态
            if hasattr(self.env, 'get_observation'):
                obs = self.env.get_observation()
                state = []
                
                # 提取末端执行器位置 (3维)
                if 'robot0_eef_pos' in obs:
                    state.extend(obs['robot0_eef_pos'][:3])
                else:
                    # 如果位置信息不可用，使用默认值
                    state.extend([0, 0, 0])
                
                # 提取末端执行器姿态 (四元数，4维)
                if 'robot0_eef_quat' in obs:
                    state.extend(obs['robot0_eef_quat'][:4])
                else:
                    # 如果四元数不可用，使用默认值 (单位四元数)
                    state.extend([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                
                # 提取夹爪状态 (1维)
                if 'robot0_gripper_qpos' in obs:
                    state.append(obs['robot0_gripper_qpos'][0])
                elif 'robot0_gripper_qpos' not in obs and 'robot0_gripper' in obs:
                    # 尝试其他可能的夹爪键名
                    state.append(obs['robot0_gripper'][0])
                else:
                    # 如果夹爪信息不可用，使用默认值
                    state.append(0.0)
                
                # print(f"[DEBUG] 获取到状态维度: {len(state)}, 内容: {state[:7]}...")  # 调试信息
                return np.array(state)
                
        except Exception as e:
            print(f"[ERROR] 获取机器人状态时出错: {e}")
        
        # 失败时返回默认的8维状态
        default_state = np.array([0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0])
        print(f"[WARN] 使用默认状态: {default_state}")
        return default_state

def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
