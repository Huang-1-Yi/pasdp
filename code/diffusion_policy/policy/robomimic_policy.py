"""Thin policy adapter around Robomimic's native image BC algorithms."""

from typing import Dict, Optional, Sequence

import torch
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.envs.robomimic.robomimic_config_util import (
    get_robomimic_config,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.common.base_policy import BasePolicy


class RobomimicImagePolicy(BasePolicy):
    """Expose native Robomimic BCRNN through this repository's policy API.

    Unlike ``rgb_model=robomimic``, this class uses Robomimic for the complete
    recurrent behavior-cloning policy, not only as an observation encoder.
    """

    def __init__(
            self,
            shape_meta: dict,
            algo_name: str = "bc_rnn",
            obs_type: str = "image",
            task_name: str = "square",
            dataset_type: str = "ph",
            crop_shape: Optional[Sequence[int]] = (76, 76)):
        super().__init__()

        action_shape = tuple(int(x) for x in shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise NotImplementedError(
                f"RobomimicImagePolicy expects one action axis, got {action_shape}"
            )

        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = {}
        for key, attr in shape_meta["obs"].items():
            obs_key_shapes[key] = list(attr["shape"])
            modality = attr.get("type", "low_dim")
            if modality not in ("rgb", "low_dim"):
                raise RuntimeError(
                    "native Robomimic image baseline supports only RGB and "
                    f"low-dimensional observations, got {key!r}: {modality!r}"
                )
            obs_config[modality].append(key)

        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type,
        )
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            for modality in config.observation.encoder.values():
                if modality.obs_randomizer_class != "CropRandomizer":
                    continue
                if crop_shape is None:
                    modality.obs_randomizer_class = None
                else:
                    crop_height, crop_width = (int(x) for x in crop_shape)
                    modality.obs_randomizer_kwargs.crop_height = crop_height
                    modality.obs_randomizer_kwargs.crop_width = crop_width

        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_shape[0],
            device="cpu",
        )
        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config
        self.algo_name = str(algo_name)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        return super().to(*args, **kwargs)

    def predict_action(
            self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_obs = self.normalizer.normalize(obs_dict)
        robomimic_obs = dict_apply(normalized_obs, lambda x: x[:, 0, ...])
        normalized_action = self.model.get_action(robomimic_obs)
        action = self.normalizer["action"].unnormalize(normalized_action)
        return {"action": action[:, None, :]}

    def reset(self) -> None:
        self.model.reset()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch: int, validate: bool = False):
        normalized_obs = self.normalizer.normalize(batch["obs"])
        normalized_action = self.normalizer["action"].normalize(batch["action"])
        robomimic_batch = {
            "obs": normalized_obs,
            "actions": normalized_action,
        }
        input_batch = self.model.process_batch_for_training(robomimic_batch)
        return self.model.train_on_batch(
            batch=input_batch,
            epoch=int(epoch),
            validate=bool(validate),
        )

    def on_epoch_end(self, epoch: int) -> None:
        self.model.on_epoch_end(int(epoch))

    def get_optimizer(self):
        return self.model.optimizers["policy"]


RobomimicBCRNNPolicy = RobomimicImagePolicy

__all__ = ["RobomimicImagePolicy", "RobomimicBCRNNPolicy"]
