# Sample and save batches
from dataloader import SplitDataloader
import numpy as np
import torch
import os, sys
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
        # A few datasets are too large. Restrict their sizes
        if ds_name in ["semeion", "libras", "arrhythmia", "miniboone"]:
            print(f"Reducing number of batches for dataset {ds_name}")
            num_batches = num_batches // 2

        dl = SplitDataloader(cfg, datasets=[ds_name], bs=num_batches, testing=True)
        raw_batch = next(iter(dl))

        batch = [torch.stack(xs) for xs in raw_batch[:-1]]
        batch.append(raw_batch[-1])

        # Some entries take too much storage space. Downsize these.
        batch[0] = batch[0].to(torch.float16)
        batch[2] = batch[2].to(torch.float16)
        batch[1] = batch[1].to(torch.int8)
        batch[3] = batch[3].to(torch.int8)



        # Save format: num_rows, num_targets, num_cols
        with open(save_file, "wb") as f:
            pickle.dump(batch, f)

    except IndexError as e:
        # print(e)
        with open(save_file, "wb") as f:
            pickle.dump(None, f)

def main():
    datasets = sorted([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')])
    #datasets = ["semeion"]
    N_batches = 2000
    for ds in datasets:
        for N_meta in [1, 2, 3, 5, 7]:
            print(f'{ds =}, {N_meta = }')

            cfg = BatchConfig(N_meta=N_meta, N_target=5)

            save_batch(cfg, ds, N_batches)


if __name__ == "__main__":
    #
    # batches = load_batch("adult", N_meta=5, N_target=5)[2][0]
    #
    # print(batches)
    main()
