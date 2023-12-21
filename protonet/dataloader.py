import torch
import numpy as np
import pandas as pd
import os
from itertools import islice
from baselines import BasicModel

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

    def __init__(self, cfg, ds_name, split, dtype=torch.float32, device="cpu"):
        self.cfg, self.RNG = cfg, cfg.RNG

        self.ds_name = ds_name
        self.device = device
        self.dtype = dtype
        # self.all_cols = all_cols

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

        for label in labels:
            mask = (position == label)
            label_data = data[mask]

            self.data[label] = label_data

        self.num_labels = len(self.data)
        self.max_labels = max(self.data.keys()) + 1  # These are different if an intermediate label is removed.

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

        if self.cfg.norm_targ:
            all_data = torch.cat([xs_meta, xs_target])
        else:
            all_data = xs_meta
        std, mean = torch.std_mean(all_data, dim=0)
        xs_meta = (xs_meta - mean) / (std + 1e-8)
        xs_target = (xs_target - mean) / (std + 1e-8)

        return xs_meta, ys_meta.to(int), xs_target, ys_target.to(int)

    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.tot_rows


class SplitDataloader:
    def __init__(self, cfg: Config, dataset, all_cols=True, device="cpu"):
        """
        :param dataset: Which datasets to sample from.
            If None: All datasets
            If -1, sample all available datasets
            If strings, sample from that specified dataset(s).
        """
        self.cfg, self.RNG = cfg, cfg.RNG
        self.all_cols = all_cols
        self.device = device

        ds_dir = f'{cfg.DS_DIR}/data/'

        try:
            self.ds = MyDataSet(cfg, dataset, device=self.device, split="all")
        except ValueError as e:
            print(f'Not sampling dataset {dataset} for reason: ')
            print(e)

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows]
        """
        while True:

            # Number of columns to sample dataset. Testing always uses full dataset
            N_cols = self.ds.tot_cols - 1
            if not self.all_cols:
                N_cols = self.RNG.choice(N_cols)  # Allow repeats.
            xs_meta, ys_meta, xs_target, ys_target = list(zip(*[
                self.ds.sample(N_cols=N_cols) for _ in range(self.cfg.bs)]))

            # Get number of labels in batch
            N_label = self.ds.max_labels
            yield xs_meta, ys_meta, xs_target, ys_target, N_label

    def __repr__(self):
        return str(self.ds)


if __name__ == "__main__":
    cfg = Config()

    dl = SplitDataloader(cfg, dataset="adult", all_cols=True)
    model = BasicModel("KNN")

    # print(dl.all_datasets[0].num_labels)
    for mp, ml, tp, tl, N_label in islice(dl, 5):
        # mp, ml, tp, tl = mp[0], ml[0], tp[0], tl[0]

        acc, _ = model.get_accuracy([mp, ml, tp, tl, None])
        print(acc)
