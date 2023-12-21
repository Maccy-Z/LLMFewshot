# Evaluate models on batches. Do the actual accuracy evaluation.

import os, toml, random, pickle, warnings
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, CatboostError

from config import Config, load_config


BASEDIR = '.'


class Model(ABC):
    max_batches = 1000

    # Process batch of data
    def get_accuracy(self, batch):
        xs_metas, ys_metas, xs_targets, ys_targets, _ = batch
        accs = []
        batch_no = 0
        for xs_meta, xs_target, ys_meta, ys_target in zip(xs_metas, xs_targets, ys_metas, ys_targets):
            self.fit(xs_meta, ys_meta)
            a = self.get_acc(xs_target, ys_target)

            accs.append(a)

            batch_no += 1
            if batch_no > self.max_batches:
                break

        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])

        return mean, std

    @abstractmethod
    def fit(self, xs_meta, ys_meta):
        pass

    @abstractmethod
    def get_acc(self, xs_target, ys_target) -> np.array:
        pass


class BasicModel(Model):
    def __init__(self, name):
        match name:
            case "LR":
                self.model = LogisticRegression(max_iter=1000)
            case "SVC":
                self.model = SVC(C=10, kernel="sigmoid", gamma=0.02)
            case "KNN":
                self.model = KNN(n_neighbors=2, p=1, weights="distance")
            case "CatBoost":
                self.model = CatBoostClassifier(iterations=100, learning_rate=0.03, allow_const_label=True, verbose=False)
            case "R_Forest":
                self.model = RandomForestClassifier(n_estimators=150, n_jobs=5)
            case _:
                raise Exception("Invalid model specified")

        self.name = name
        self.identical_batch = False

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten().numpy()
        xs_meta = xs_meta.numpy()

        if ys_meta.min() == ys_meta.max():
            self.identical_batch = True
            self.pred_val = ys_meta[0]
        else:
            self.identical_batch = False

            try:
                self.model.fit(xs_meta, ys_meta)
            except CatboostError:
                # Catboost fails if every input element is the same
                self.identical_batch = True
                mode = stats.mode(ys_meta, keepdims=False)[0]
                self.pred_val = mode

    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target.numpy()
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            predictions = self.model.predict(xs_target)

        return np.array(predictions).flatten() == np.array(ys_target)

    def __repr__(self):
        return self.name


#
# class FLAT_MAML(Model):
#     def __init__(self, load_no, save_ep=None):
#         save_dir = f'{BASEDIR}/saves/save_{load_no}'
#         print(f'Loading model at {save_dir = }')
#
#         if save_ep is None:
#             state_dict = torch.load(f'{save_dir}/model.pt')
#         else:
#             state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')
#         self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
#         self.model.load_state_dict(state_dict['model_state_dict'])
#
#     def fit(self, xs_meta, ys_meta):
#         xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
#         pairs_meta = d2v_pairer(xs_meta, ys_meta)
#         with torch.no_grad():
#             embed_meta, pos_enc = self.model.forward_meta(pairs_meta)
#
#         embed_meta.requires_grad = True
#         pos_enc.requires_grad = True
#         optim_pos = torch.optim.Adam([pos_enc], lr=0.001)
#         # optim_embed = torch.optim.SGD([embed_meta, ], lr=50, momentum=0.75)
#         optim_embed = torch.optim.Adam([embed_meta], lr=0.075)
#         for _ in range(5):
#             # Make predictions on meta set and calc loss
#             preds = self.model.forward_target(xs_meta, embed_meta, pos_enc)
#             loss = torch.nn.functional.cross_entropy(preds.squeeze(), ys_meta.long().squeeze())
#             loss.backward()
#             optim_pos.step()
#             optim_embed.step()
#             optim_embed.zero_grad()
#             optim_pos.zero_grad()
#
#         self.embed_meta = embed_meta
#         self.pos_enc = pos_enc
#
#         # print(self.embed_meta)
#
#     def get_acc(self, xs_target, ys_target) -> np.array:
#         xs_target = xs_target.unsqueeze(0)
#         with torch.no_grad():
#             ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)
#
#         ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
#
#         return (ys_pred_target_labels == ys_target).numpy()
#
#     def __repr__(self):
#         return "FLAT_maml"


def get_results_by_dataset(test_data_names, models, N_meta=10, N_target=5):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    results = pd.DataFrame(columns=['data_name', 'model', 'acc', 'std'])

    # Test on full dataset
    for data_name in test_data_names:
        print(data_name)
        try:
            batch = load_batch(ds_name=data_name, N_meta=N_meta, N_target=N_target)
        except IndexError as e:
            print(e)
            continue

        model_acc_std = defaultdict(list)
        for model in models:
            mean_acc, std_acc = model.get_accuracy(batch)

            model_acc_std[str(model)].append([mean_acc, std_acc])

            # print(model.preds)
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
                'acc': mean_acc,
                'std': std_acc
            }, index=[0])
            results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def process_results(test_results):
    # Results for each dataset
    detailed_results = test_results.copy()

    mean, std = detailed_results["acc"], detailed_results["std"]
    mean_std = [f'{m * 100:.2f}±{s * 100:.2f}' for m, s in zip(mean, std)]
    detailed_results['acc_std'] = mean_std

    results = detailed_results.pivot(columns=['data_name'], index='model', values=['acc_std'])
    print("\n======================================================")
    print("Accuracy")
    print(results.to_string())

    det_results = detailed_results.pivot(columns=['data_name'], index='model', values=['acc'])
    det_results = det_results.to_string()

    # Aggreate results
    agg_results = test_results.copy()
    print(test_results)

    def combine_stddev(series):
        return np.sqrt((series ** 2).sum()) / len(series)

    # Move flat to first column
    mean_acc = agg_results.groupby('model')['acc'].mean()
    std_acc = agg_results.groupby('model')['std'].agg(combine_stddev)
    mean_acc = pd.concat([mean_acc, std_acc], axis=1)

    best_baseline_acc = mean_acc.drop("FLAT").max()

    flat_diff = mean_acc.loc["FLAT"]["acc"] - best_baseline_acc["acc"]
    flat_diff = pd.DataFrame({"acc": flat_diff, 'std': None}, index=['FLAT diff'])

    mean_acc = pd.concat([flat_diff, mean_acc])

    # print()
    print()
    print("======================================================")
    print("Accuracy (aggregated)")
    print(mean_acc.to_string())

    exit(5)

    new_column_order = ["FLAT", "FLAT_maml"] + [col for col in agg_results.columns if (col != "FLAT" and col != "FLAT_maml")]
    agg_results = agg_results.reindex(columns=new_column_order)

    # Difference between FLAT and best model
    agg_results["FLAT_diff"] = (agg_results["FLAT"] - agg_results.iloc[:, 2:].max(axis=1)) * 100
    agg_results["FLAT_maml_diff"] = (agg_results["FLAT_maml"] - agg_results.iloc[:, 2:-1].max(axis=1)) * 100
    agg_results["FLAT_diff"] = agg_results["FLAT_diff"].apply(lambda x: f'{x:.2f}')
    agg_results["FLAT_maml_diff"] = agg_results["FLAT_maml_diff"].apply(lambda x: f'{x:.2f}')

    # Get errors using appropriate formulas.
    pivot_acc = test_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['acc'])
    pivot_std = test_results.pivot(
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

    print()
    print("======================================================")
    print("Accuracy (aggregated)")
    print(agg_results["FLAT_diff"].to_string(index=False))
    # print(agg_results.to_string(index=False))
    print(agg_results.to_string())
    agg_results = agg_results.to_string()

    # with open(f'{result_dir}/aggregated', "w") as f:
    #     for line in agg_results:
    #         f.write(line)
    #
    # with open(f'{result_dir}/detailed', "w") as f:
    #     for line in det_results:
    #         f.write(line)
    #
    # with open(f'{result_dir}/raw.pkl', "wb") as f:
    #     pickle.dump(unseen_results, f)


def main(load_no, N_meta):
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    load_no = [existing_saves[num] for num in load_no]

    load_dir = f'{BASEDIR}/saves/save_{load_no[-1]}'

    # result_dir = f'{BASEDIR}/Results'
    # files = [f for f in os.listdir(result_dir) if os.path.isdir(f'{result_dir}/{f}')]
    # existing_results = sorted([int(f) for f in files if f.isdigit()])
    #
    # print(existing_results)
    # result_no = existing_results[-1] + 1
    #
    # result_dir = f'{result_dir}/{result_no}'
    # print(result_dir)
    # os.mkdir(result_dir)

    split_name = "0"
    splits = toml.load(f'./datasets/splits/{split_name}')

    test_datasets = splits["test"]
    test_datasets.remove("semeion")
    # print("Train datases:", train_data_names)
    print("Test datasets:", test_datasets)

    # test_datasets = ["adult"]

    N_target = 5
    cfg = load_config(f'{load_dir}/config.toml')  # Config()
    cfg.N_meta = N_meta
    cfg.N_target = N_target

    models = [FLAT(num) for num in load_no] + [BasicModel("LR"), BasicModel("R_Forest")]

    test_results = get_results_by_dataset(test_datasets, models, N_meta=N_meta, N_target=N_target)

    process_results(test_results)


