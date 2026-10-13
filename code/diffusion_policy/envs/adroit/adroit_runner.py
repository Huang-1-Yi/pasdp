import wandb
import time
import numpy as np
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.envs.adroit.adroit import AdroitEnv
from diffusion_policy.envs.base_image_runner import BaseImageRunner
from diffusion_policy.envs.gym_util.mjpc_diffusion_wrapper import (
    MujocoPointcloudWrapperAdroit,
)
from diffusion_policy.envs.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.envs.gym_util.scalar_recorder import LargestKRecorder
from diffusion_policy.envs.gym_util.simple_video_recording_wrapper import (
    SimpleVideoRecordingWrapper,
)
from diffusion_policy.policy.common.base_policy import BasePolicy
from termcolor import cprint


class AdroitRunner(BaseImageRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=20,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = LargestKRecorder(k=3)
        self.logger_util_test10 = LargestKRecorder(k=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_time = []
        
         
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            is_success = False
            actual_step_count = 0
            total_time = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    start_time = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    end_time = time.time()
                    total_time += end_time - start_time
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['goal_achieved'])
                # MultiStepWrapper keeps only the latest observation window in
                # info. Preserve success over the complete episode.
                is_success = is_success or bool(np.any(info['goal_achieved']))
                done = np.all(done)
                actual_step_count += 1

             # print(total_time / actual_step_count)
            all_success_rates.append(is_success)
            all_goal_achieved.append(num_goal_achieved)
            all_time.append(total_time / max(actual_step_count, 1))


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_time'] = np.mean(all_time)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average()
        log_data['SR_test_L5'] = self.logger_util_test10.average()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        return log_data

    def close(self):
        env, self.env = self.env, None
        if env is not None:
            env.close()
