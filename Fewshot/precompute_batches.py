# Sample and save batches
from dataloader import SplitDataloader
import numpy as np
import torch
import os
import pickle

data_dir = "./datasets/data"


# Load a batch from disk
def load_batch(ds_name, N_meta, N_target):
    with open(f"./datasets/data/{ds_name}/batches/meta_{N_meta}_targ_{N_target}", "rb") as f:
        batch = pickle.load(f)
    if batch is None:
        raise IndexError(f"Batch not found for file {ds_name}")

    return batch


# Fixed config file used for generating batches
class BatchConfig:
    # Dataloader properties
    min_row_per_label: int = 20  # Minimum number of rows of a label
    min_cols: int = 3  # Minimum number of dataset columns

    fix_per_label: bool = True  # Fix N_meta per label instead of total
    N_meta: int  # N rows in meta
    N_target: int  # N rows in target

    col_fmt: str = 'uniform'  # How to sample number of columns per batch
    normalise: bool = True  # Normalise predictors

    # Train DL params
    DS_DIR: str = "./datasets"
    # ds_group: str   # Datasets to sample from. List or filename
    bs: int = 3

    # RNG
    seed: int = 0

    def __init__(self, N_meta, N_target):
        self.N_meta = N_meta
        self.N_target = N_target
        assert self.min_row_per_label >= self.N_meta + self.N_target

        self.RNG = np.random.default_rng(seed=self.seed)
        self.T_RNG = torch.Generator()
        self.T_RNG.manual_seed(self.seed)


# Save a batch to disk
def save_batch(cfg, ds_name, num_batches):
    save_file = f"{data_dir}/{ds_name}/batches/meta_{cfg.N_meta}_targ_{cfg.N_target}"

    if not os.path.exists(f"{data_dir}/{ds_name}/batches"):
        os.makedirs(f"{data_dir}/{ds_name}/batches")

    try:
        dl = SplitDataloader(cfg, datasets=[ds_name], bs=num_batches, testing=True)
        batch = next(iter(dl))
        # Save format: num_rows, num_targets, num_cols
        with open(save_file, "wb") as f:
            pickle.dump(batch, f)

    except IndexError as e:
        print(e)
        with open(save_file, "wb") as f:
            pickle.dump(None, f)


def main():
    datasets = sorted([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')])

    N_batches = 2000
    for ds in datasets:
        for N_meta in [1, 2, 3, 5, 7]:
            cfg = BatchConfig(N_meta=N_meta, N_target=5)

            save_batch(cfg, ds, N_batches)


if __name__ == "__main__":
    #
    # batches = load_batch("adult", N_meta=5, N_target=5)[2][0]
    #
    # print(batches)
    main()
