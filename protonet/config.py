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
    N_meta: int = 128  # N rows in meta
    N_target: int = 10  # N rows in target

    norm_targ: bool = False  # Normalise predictors

    # Eval params
    DS_DIR: str = './datasets'
    ds_group: str = '0'  # Datasets to sample from. List or filename
    bs: int = 256     # Batch size
    N_batches: int = 8 # Number of batches to sample

    # Model parameters
    proto_dim: int = 256

    # RNGs
    seed: int = 0

    def __post_init__(self):
        self.T_RNG = torch.Generator()

        if self.seed is None:
            self.T_RNG.seed()
            self.RNG = np.random.default_rng()
        else:
            self.T_RNG.manual_seed(self.seed)
            self.RNG = np.random.default_rng(self.seed)


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


