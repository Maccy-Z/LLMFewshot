# Get accuracies from precomputed models

from main import *
import os, toml, random
import numpy as np
from abc import ABC
import pandas as pd
from collections import defaultdict

from precompute_batches import load_batch

import sys

sys.path.append('/mnt/storage_ssd/FairFewshot/STUNT_main')
# from STUNT_interface import STUNT_utils, MLPProto

BASEDIR = '.'


class PreModel(ABC):
    # Process batch of data
    def __init__(self, model_name):
        self.model_name = model_name

    def get_accuracy(self, ds_name, num_rows):
        with open(f'./datasets/data/{ds_name}/baselines.dat', "r") as f:
            lines = f.read()

        lines = lines.split("\n")

        for config in lines:
            if config.startswith(f'{self.model_name},{num_rows}'):
                config = config.split(",")

                mean, std = float(config[-2]), float(config[-1])
                return mean, std

        raise FileNotFoundError(f"Requested config does not exist: {self.model_name}, {ds_name}, {num_rows=}")

    def __repr__(self):
        return self.model_name


def get_results_by_dataset(test_data_names, models, N_meta):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc', 'std'])

    # Test on full dataset
    for data_name in test_data_names:
        model_acc_std = defaultdict(list)
        for model in models:
            try:
                mean_acc, std_acc = model.get_accuracy(data_name, N_meta)
            except FileNotFoundError as e:
                print(e)
                continue

            model_acc_std[str(model)].append([mean_acc, std_acc])

        for model_name, acc_stds in model_acc_std.items():
            acc_stds = np.array(acc_stds)
            # For baselines, variance is sample variance.
            if len(acc_stds) == 1:
                mean_acc, std_acc = acc_stds[0, 0], acc_stds[0, 1]

            # Average over all FLAT and FLAT_MAML models.
            # For FLAT, variance is variance between models
            else:
                means, std = acc_stds[:, 0], acc_stds[:, 1]
                mean_acc = np.mean(means)
                std_acc = np.std(means, ddof=1) / np.sqrt(means.shape[0])

            result = pd.DataFrame({
                'data_name': data_name,
                'model': str(model_name),
                'num_cols': -1,
                'acc': mean_acc,
                'std': std_acc
            }, index=[0])
            results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def main(load_no, num_rows, save_ep=None, num_1s=None):
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    load_no = [existing_saves[num] for num in load_no]
    load_dir = f'{BASEDIR}/saves/save_{load_no[-1]}'

    split_name = "0"
    splits = toml.load(f'./datasets/splits/{split_name}')

    test_splits = splits["test"]
    print(splits)
    print("Testing group:", split_name)

    N_target = 5

    models = [PreModel("LR"), PreModel("R_Forest"), PreModel("CatBoost"),
              ]

    unseen_results = get_results_by_dataset(
        test_splits, models, N_meta=num_rows)

    # Results for each dataset
    detailed_results = unseen_results.copy()

    mean, std = detailed_results["acc"], detailed_results["std"]
    mean_std = [f'{m * 100:.2f}±{s * 100:.2f}' for m, s in zip(mean, std)]
    detailed_results['acc_std'] = mean_std

    results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc_std'])
    # print("======================================================")
    # print("Test accuracy on unseen datasets")
    # print(results.to_string())

    det_results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc'])
    det_results = det_results.to_string()

    # Aggreate results
    agg_results = unseen_results.copy()

    # Move flat to first column
    agg_results = agg_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()

    # Get errors using appropriate formulas.
    pivot_acc = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['acc'])
    pivot_std = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['std'])
    model_names = pivot_acc.columns.get_level_values(2).unique()
    for model_name in model_names:

        model_accs = pivot_acc.loc[:, ("acc", slice(None), model_name)]
        model_stds = pivot_std.loc[:, ("std", slice(None), model_name)]

        mean_stds = []
        for i in range(pivot_acc.shape[0]):
            accs = np.array(model_accs.iloc[i].dropna())
            std = np.array(model_stds.iloc[i].dropna())

            assert std.shape == accs.shape
            mean_acc = np.mean(accs)
            std_acc = np.sqrt(np.sum(std ** 2)) / std.shape[0]
            mean_std = f'{mean_acc * 100:.2f}±{std_acc * 100:.2f}'
            mean_stds.append(mean_std)

        agg_results[model_name] = mean_stds

    # print()
    # print("======================================================")
    # print("Test accuracy on unseen datasets (aggregated)")
    # print(agg_results.to_string(index=False))
    print(agg_results.to_string())


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    for i in [0, 1, 2, 3]:
        col_accs = main(load_no=[0], num_rows=10, save_ep=[i, -1])
