"""ACT network and optimizer construction."""

import torch

from .models import build_ACT_model


class Config:
    """Attribute view over ACT's upstream dictionary-style configuration."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def build_ACT_model_and_optimizer(args_override):
    args = Config({
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "backbone": "resnet18",
        "dilation": False,
        "position_embedding": "sine",
        "camera_names": [],
        "enc_layers": 4,
        "dec_layers": 6,
        "dim_feedforward": 2048,
        "hidden_dim": 256,
        "dropout": 0.1,
        "nheads": 8,
        "num_queries": 400,
        "pre_norm": False,
        "masks": False,
    })
    for key, value in args_override.items():
        setattr(args, key, value)

    model = build_ACT_model(args)
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" not in name and parameter.requires_grad
            ],
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" in name and parameter.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return model, optimizer


__all__ = ["build_ACT_model_and_optimizer"]
