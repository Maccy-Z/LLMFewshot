import torch.random

import toml
import dataclasses
from dataclasses import asdict, fields
import numpy as np
import toml


@dataclasses.dataclass
class Config:
    # Dataloader properties
    min_row_per_label: int = 20  # Minimum number of rows of a label
    min_cols: int = 3  # Minimum number of dataset columns
    max_labels: int = 10

    fix_per_label: bool = True  # Fix N_meta per label instead of total
    N_meta: int = 5  # N rows in meta
    N_target: int = 5  # N rows in target

    col_fmt: str = 'uniform'  # How to sample number of columns per batch
    normalise: bool = True  # Normalise predictors

    # Train DL params
    DS_DIR: str = './datasets'
    ds_group: str = '0'  # Datasets to sample from. List or filename
    bs: int = 8

    # Model parameters
    proto_dim: int = 16

    d2v_h_dim: int = 64
    f_depth: int = 4
    pos_enc_bias: str = "zero"
    pos_enc_dim: int = 15
    pos_enc_depth: int = 2

    gat_heads: int = 1
    gat_hid_dim: int = 64
    gat_in_dim: int = pos_enc_dim + 1
    gat_out_dim: int = 16
    gat_layers: int = 2

    # RNGs
    seed: int = None

    # Optimiser parameters

    lr: float = 5e-4  # 5e-4
    eps: float = 3e-4  # 3e-4
    w_decay: float = 1e-4

    # Training duration
    epochs: int = 31
    ep_len: int = 2000
    val_len: int = 500

    def __post_init__(self):
        assert self.min_row_per_label >= self.N_meta + self.N_target

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


