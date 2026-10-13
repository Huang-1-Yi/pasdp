if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_padp_workspace import (
    TrainPADPWorkspace,
)


class TrainPADPUnrolledWorkspace(TrainPADPWorkspace):
    """PADP variant that enables the self-rollout curriculum."""

    def _rollout_enabled(self, cfg: OmegaConf) -> bool:
        return bool(getattr(cfg.training, "enable_rollout", True))

    def _on_epoch_start(self, cfg: OmegaConf) -> None:
        if hasattr(self.model, "_set_self_rollout_prob"):
            self.model._set_self_rollout_prob(
                self.epoch, cfg.training.num_epochs
            )


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainPADPUnrolledWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
