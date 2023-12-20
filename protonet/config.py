import torch.random

import toml
import dataclasses
from dataclasses import asdict, fields
import numpy as np
import toml


@dataclasses.dataclass
class Config:
    # Dataloader properties
    fix_per_label: bool = True  # Fix N_meta per label instead of total
    N_meta: int = 5  # N rows in meta
    N_target: int = 5  # N rows in target

    normalise: bool = True  # Normalise predictors

    # Train DL params
    DS_DIR: str = './datasets'
    ds_group: str = '0'  # Datasets to sample from. List or filename
    bs: int = 8
    N_batches: int = 100

    # Model parameters
    proto_dim: int = 256

    # RNGs
    seed: int = None

    # Optimiser parameters
    lr: float = 3e-4  # 5e-4
    eps: float = 0.1e-4  # 3e-4
    w_decay: float = 1e-4

    def __post_init__(self):
        self.RNG = np.random.default_rng()
        self.T_RNG = torch.Generator()
        #self.T_RNG.manual_seed()


def save_config(cfg: Config, save_file):
    with open(save_file, "w") as f:
        toml.dump(asdict(cfg), f)


def load_config(save_file):
    with open(save_file, "r") as f:
        config = toml.load(f)

    for field in fields(Config()):
        if field.name not in config:
            config[field.name] = field.default

    return Config(**config)


