import torch
import numpy as np
import pandas as pd
import os
from itertools import islice
import toml

from config import Config


def d2v_pairer(batch_xs, batch_ys):
    # xs.shape = [bs][N_meta, N_cols], ys.shape = [bs][N_meta]

    batch_pairs = []
    for xs, ys in zip(batch_xs, batch_ys):
        N_meta, N_cols = xs.shape
        # xs = xs.reshape(bs * N_meta, N_cols)
        # ys = ys.reshape(bs * N_meta, 1)

        pair_flat = torch.empty(N_meta, N_cols, 2)
        for k, (xs_k, ys_k) in enumerate(zip(xs, ys)):
            # Only allow 1D for ys
            ys_k = ys_k.repeat(N_cols)
            pairs = torch.stack([xs_k, ys_k], dim=-1)

            pair_flat[k] = pairs

        batch_pairs.append(pair_flat)
    return batch_pairs


def to_tensor(array: np.array, device=torch.device('cpu'), dtype=torch.float32):
    return torch.from_numpy(array).to(device).to(dtype)


# Sample n items from k catagories. Return samples per catagory.
def sample(n, k):
    q, r = divmod(n, k)
    counts = [q + 1] * r + [q] * (k - r)
    return counts


class MyDataSet:
    cfg: Config

    def __init__(self, cfg, ds_name, split, testing, dtype=torch.float32, device="cpu"):
        self.cfg, self.RNG = cfg, cfg.RNG

        self.ds_name = ds_name
        self.device = device
        self.dtype = dtype
        self.testing = testing

        """
        Dataset format: {folder}_py.dat             predictors
                        labels_py.dat               labels for predictors
                        folds_py.dat                test fold
                        validation_folds_py.dat     validation fold
        folds_py == 0 for train
        vfold_py == 1 for validation
        folds_py == 1 for test                 
        
        Here, combine test and valid folds. 
        """

        ds_dir = f'{cfg.DS_DIR}/data'
        # get train fold
        folds = pd.read_csv(
            f"{ds_dir}/{ds_name}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(
            f"{ds_dir}/{ds_name}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(
            f"{ds_dir}/{ds_name}/{self.ds_name}_py.dat", header=None)
        predictors = np.asarray(predictors)
        # read internal target
        targets = pd.read_csv(f"{ds_dir}/{ds_name}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        if split == "train":
            idx = (1 - folds) == 1 & (vldfold == 0)
        elif split == "val":
            idx = (vldfold == 1)
        elif split == "test":
            idx = (folds == 1)
        elif split == "all":
            idx = np.ones_like(folds).astype(bool)
        else:
            raise Exception("Split must be train, val, test or all")

        preds = predictors[idx]
        labels = targets[idx]

        data = np.concatenate((preds, labels), axis=-1)
        data = to_tensor(data)
        self.tot_rows, self.tot_cols = data.shape[0], data.shape[-1]

        labels, position = np.unique(data[:, -1], return_inverse=True)
        labels = labels.astype(int)

        # Sort data by label
        self.data = {}

        # Include labels that have enough entries. If there are less than 2 valid labels, discard dataset.
        for label in labels:
            mask = (position == label)
            label_data = data[mask]

            num_label_row = len(label_data)
            if cfg.min_row_per_label > num_label_row:
                print(f'Not enough labels for {self}, class {label}, require {cfg.min_row_per_label}, has {num_label_row}')
            else:
                self.data[label] = label_data

        self.num_labels = len(self.data)
        self.max_labels = max(self.data.keys()) + 1  # These are different if an intermediate label is removed.

        if self.num_labels < 2:
            raise ValueError(f'Not enough labels. {self.num_labels} labels for dataset {self.ds_name}')

        if self.tot_cols < cfg.min_cols:
            raise ValueError(f'Not enough columns. Require {cfg.min_cols}, has {self.tot_cols}')

    def sample(self, N_cols):
        # Columns to sample from
        pred_cols = self.RNG.choice(self.tot_cols - 1, size=N_cols, replace=False)

        if self.cfg.fix_per_label:
            sample_meta = [self.cfg.N_meta for _ in range(self.num_labels)]
            sample_target = [self.cfg.N_target for _ in range(self.num_labels)]
        # Uniformly divide labels to fit n_meta / target.
        else:
            sample_meta = self.RNG.permutation(sample(self.cfg.N_meta, self.num_labels))
            sample_target = self.RNG.permutation(sample(self.cfg.N_target, self.num_labels))

        # Some datasets have too many labels
        if not self.testing:
            wanted_labels = self.RNG.permutation(list(self.data.keys()))[:self.RNG.integers(2, self.cfg.max_labels)]
        else:
            wanted_labels = list(self.data.keys())

        # Draw number of samples from each label.
        metas, targets = [], []
        for (label, label_rows), N_meta, N_target in zip(self.data.items(), sample_meta, sample_target, strict=True):
            if label in wanted_labels:
                # Draw rows and shuffle to make meta and target batch
                idx = torch.randperm(label_rows.size(0), generator=self.cfg.T_RNG)[: N_meta + N_target]
                rows = label_rows[idx]
                meta_rows = rows[:N_meta]
                target_rows = rows[N_meta:]

                metas.append(meta_rows)
                targets.append(target_rows)

        metas, targets = torch.cat(metas), torch.cat(targets)
        ys_meta, ys_target = metas[:, -1], targets[:, -1]
        xs_meta, xs_target = metas[:, pred_cols], targets[:, pred_cols]

        if self.cfg.normalise:
            all_data = torch.cat([xs_meta, xs_target])
            std, mean = torch.std_mean(all_data, dim=0)
            xs_meta = (xs_meta - mean) / (std + 1e-8)
            xs_target = (xs_target - mean) / (std + 1e-8)

        return xs_meta, ys_meta.to(int), xs_target, ys_target.to(int)

    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.tot_rows


class SplitDataloader:
    def __init__(self, cfg, bs, datasets, testing, device="cpu"):
        """
        :param bs: Number of datasets to sample from each batch
        :param datasets: Which datasets to sample from.
            If None: All datasets
            If -1, sample all available datasets
            If strings, sample from that specified dataset(s).
        :param ds_split: If ds_group is int >= 0, the test or train split.
        """
        assert isinstance(testing, bool)
        self.cfg, self.RNG = cfg, cfg.RNG
        self.bs = bs
        self.testing = testing
        self.device = device

        ds_dir = f'{cfg.DS_DIR}/data/'

        # All datasets
        if datasets is None:
            ds_names = [f for f in os.listdir(ds_dir) if os.path.isdir(f'{ds_dir}/{f}')]
            print(ds_names)
        # Specific datasets
        elif isinstance(datasets, list):
            ds_names = datasets
        # Premade splits
        elif isinstance(datasets, str):
            with open(f'{cfg.DS_DIR}/splits/{datasets}', "r") as f:
                split = "test" if testing else "train"
                ds_names = toml.load(f)[split]
        else:
            raise Exception("Invalid ds_group")

        self.all_datasets = []
        for ds_name in ds_names:
            try:
                ds = MyDataSet(cfg, ds_name, testing=self.testing, device=self.device, split="all")
                self.all_datasets.append(ds)

            except ValueError as e:
                print(f'Discarding dataset {ds_name}')
                print(e)

        if len(self.all_datasets) == 0:
            raise IndexError(f"No datasets with enough rows")

        self.min_ds_cols = min([ds.tot_cols for ds in self.all_datasets])

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:

            # Number of columns to sample dataset. Testing always uses full dataset
            if self.cfg.col_fmt == 'all' or self.testing:
                sample_ds = self.RNG.choice(self.all_datasets, size=self.bs)  # Allow repeats.
                N_cols = min([d.tot_cols for d in sample_ds]) - 1

            elif self.cfg.col_fmt == 'uniform':
                sample_ds = self.RNG.choice(self.all_datasets, size=self.bs)  # Allow repeats.
                max_num_cols = min([d.tot_cols for d in sample_ds])# - 1
                #print(max_num_cols)
                N_cols = self.RNG.integers(3, max_num_cols)

            else:
                raise Exception("Invalid num_cols")

            xs_meta, ys_meta, xs_target, ys_target = list(zip(*[
                ds.sample(N_cols=N_cols) for ds in sample_ds]))

            # Get maximum number of labels in batch
            max_N_label = max([d.max_labels for d in sample_ds])
            yield xs_meta, ys_meta, xs_target, ys_target, max_N_label

    def __repr__(self):
        return str(self.all_datasets)


if __name__ == "__main__":
    # torch.manual_seed(0)
    cfg = Config()
    cfg.max_labels = 2
    RNG = cfg.RNG

    dl = SplitDataloader(cfg, bs=1, datasets=["abalone"], testing=True)

    # print(dl.all_datasets[0].num_labels)

    for mp, ml, tp, tl, datanames in islice(dl, 1):
        mp, ml = torch.stack(mp), torch.stack(ml)
        tp, tl = torch.stack(tp), torch.stack(tl)
        print(mp.shape, ml.shape)
        print(tp.shape, tl.shape)
        print()
