import gym
import numpy as np
from termcolor import cprint
from diffusion_policy.envs.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy.envs.gym_util.point_sampling import (
    point_cloud_sampling as _point_cloud_sampling,
)
from typing import NamedTuple, Any
from dm_env import StepType

ADROIT_PC_TRANSFORM = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(45)), np.sin(np.radians(45))],
                    [0, -np.sin(np.radians(45)), np.cos(np.radians(45))]])

ADROIT_HAND_FINGER_NAMES = {'palm',
            'ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal',
            'mfknuckle', 'mfproximal', 'mfmiddle', 'mfdistal',
            'rfknuckle', 'rfproximal', 'rfmiddle', 'rfdistal',
            'lfknuckle', 'lfproximal', 'lfmiddle', 'lfdistal',
            'thbase', 'thproximal', 'thmiddle', 'thdistal',}

ENV_POINT_CLOUD_CONFIG = {
    
    'adroit_hammer': {
        'min_bound': [-10, -10, -0.099],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_door': {
       'min_bound': [-10, -10, -0.499],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_pen': {
        'min_bound': [-10, -10, -0.79],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['vil_camera'],
        'transform': None,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 0.]),
    },

}

def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='uniform'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb
    """
    if num_points == 'all':
        return point_cloud
    return _point_cloud_sampling(
        point_cloud,
        num_points=int(num_points),
        method=method,
        fps_backend='open3d',
    )
    

class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    observation_pointcloud: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

    
class MujocoPointcloudWrapperAdroit(gym.Wrapper):
    """
    fetch point cloud from mujoco and add it to obs
    """
    def __init__(self, env, env_name:str, use_point_crop=True):
        super().__init__(env)
        self.env_name = env_name
        # point cloud cropping
        self.min_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('min_bound', None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('max_bound', None)
        
        self.use_point_crop = use_point_crop
        cprint(f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", 'green')

        
        # point cloud sampling
        self.num_points = ENV_POINT_CLOUD_CONFIG[env_name].get('num_points', 512)
        self.point_sampling_method = ENV_POINT_CLOUD_CONFIG[env_name].get('point_sampling_method', 'uniform')
        cprint(f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
        assert self.point_sampling_method in ['uniform', 'fps'], \
            f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
        
        # point cloud generator
        self.pc_generator = PointCloudGenerator(sim=env.get_mujoco_sim(),
                                                cam_names=ENV_POINT_CLOUD_CONFIG[env_name]['cam_names'])
        self.pc_transform = ENV_POINT_CLOUD_CONFIG[env_name].get('transform', None)
        self.pc_scale = ENV_POINT_CLOUD_CONFIG[env_name].get('scale', None)
        self.pc_offset = ENV_POINT_CLOUD_CONFIG[env_name].get('offset', None)

        observation_spaces = dict(env.observation_space.spaces)
        observation_spaces['point_cloud'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_points, 3),
            dtype=np.float32,
        )
        observation_spaces['depth'] = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(84, 84),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(observation_spaces)
    
    

    def get_point_cloud(self, use_RGB=False):
        save_img_dir = None
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(save_img_dir=save_img_dir) # (N, 6), xyz+rgb
        
        
        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
            
        # sampling to fixed number of points
        point_cloud = point_cloud_sampling(point_cloud=point_cloud,
                                           num_points=self.num_points, 
                                           method=self.point_sampling_method)
        
        if not use_RGB:
            point_cloud = point_cloud[:, :3]
        return (
            point_cloud.astype(np.float32, copy=False),
            depth.astype(np.float32, copy=False),
        )


    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        point_cloud, depth = self.get_point_cloud()
        
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()
        point_cloud, depth = self.get_point_cloud()
        obs_dict['point_cloud'] = point_cloud
        obs_dict['depth'] = depth
        return obs_dict
