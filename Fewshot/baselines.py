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
# from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, CatboostError
from tab_transformer_pytorch import FTTransformer

from main import *
from config import Config, load_config
from precompute_batches import load_batch

# sys.path.append('/mnt/storage_ssd/fewshot_learning/FairFewshot/STUNT_main')
# from STUNT_interface import STUNT_utils, MLPProto

BASEDIR = '.'
max_batches = 40


class Model(ABC):
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
            if batch_no > max_batches:
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


#
# class STUNT(STUNT_utils, Model):
#     model: torch.nn.Module
#
#     def __init__(self):
#         self.lr = 0.0001
#         self.model_size = (1024, 1024)  # num_cols, out_dim, hid_dim
#         self.steps = 5
#         self.tasks_per_batch = 4
#         self.test_num_way = 2
#         self.query = 1
#         self.kmeans_iter = 5
#
#     def fit(self, xs_meta, ys_meta):
#         self.shot = (xs_meta.shape[0] - 2) // 2
#         ys_meta = ys_meta.flatten().long()
#
#         # Reset the model
#         self.model = MLPProto(xs_meta.shape[-1], self.model_size[0], self.model_size[1])
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         with warnings.catch_warnings():
#             # warnings.simplefilter("ignore")
#             for _ in range(self.steps):
#                 try:
#                     train_batch = self.get_batch(xs_meta.clone())
#                     self.protonet_step(train_batch)
#                 except NameError as e:
#                     pass
#
#         with torch.no_grad():
#             meta_embed = self.model(xs_meta)
#
#         self.prototypes = self.get_prototypes(meta_embed.unsqueeze(0), ys_meta.unsqueeze(0), 2)
#
#     def get_acc(self, xs_target, ys_target):
#         self.model.eval()
#         with torch.no_grad():
#             support_target = self.model(xs_target)
#
#         self.prototypes = self.prototypes[0]
#         support_target = support_target.unsqueeze(1)
#
#         sq_distances = torch.sum((self.prototypes
#                                   - support_target) ** 2, dim=-1)
#
#         # print(sq_distances.shape)
#         _, preds = torch.min(sq_distances, dim=-1)
#
#         # print(preds.numpy(), ys_target.numpy())
#         return (preds == ys_target).numpy()
#
#
# class TabnetModel(Model):
#     def __init__(self):
#         self.model = TabNetClassifier(device_name="cpu")
#         self.bs = 64
#         self.patience = 17
#
#     def fit(self, xs_meta, ys_meta):
#         ys_meta = ys_meta.flatten().float().numpy()
#         xs_meta = xs_meta.numpy()
#
#         if ys_meta.min() == ys_meta.max():
#             self.identical_batch = True
#             self.pred_val = ys_meta[0]
#         else:
#             self.identical_batch = False
#
#             sys.stdout = open(os.devnull, "w")
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#
#                 try:
#                     # self.model.fit(xs_meta, ys_meta, drop_last=False)
#                     self.model.fit(xs_meta, ys_meta,
#                                    eval_name=["accuracy"], eval_set=[(xs_meta, ys_meta)],
#                                    batch_size=self.bs, patience=self.patience, drop_last=False)
#                 except RuntimeError:
#                     # Tabnet fails if multiple columns are exactly identical. Add a irrelevant amount of random noise to stop this.
#                     xs_meta += np.random.normal(size=xs_meta.shape) * 1e-6
#                     print(xs_meta)
#                     self.model.fit(xs_meta, ys_meta,
#                                    eval_set=[(xs_meta, ys_meta)], eval_name=["accuracy"],
#                                    batch_size=self.bs, patience=self.patience, drop_last=False)
#
#             sys.stdout = sys.__stdout__
#
#             self.pred_val = None
#
#     def get_acc(self, xs_target, ys_target):
#         if self.identical_batch:
#             predictions = np.ones_like(ys_target) * self.pred_val
#         else:
#             with torch.no_grad():
#                 predictions = self.model.predict(X=xs_target)
#
#         ys_lr_target_labels = np.array(predictions).flatten()
#
#         return ys_lr_target_labels == np.array(ys_target)
#
#     def __repr__(self):
#         return "TabNet"
#
#
# class FTTrModel(Model):
#     model: torch.nn.Module
#
#     def __init__(self):
#         self.null_categ = torch.tensor([[]])
#
#     def fit(self, xs_meta, ys_meta):
#         ys_meta = ys_meta.flatten().long()
#         xs_meta = xs_meta
#         # Reset the model
#         self.model = FTTransformer(
#             categories=(),  # tuple containing the number of unique values within each category
#             num_continuous=xs_meta.shape[-1],  # number of continuous values
#             dim=24,  # dimension, paper set at 32
#             dim_out=2,  # binary prediction, but could be anything
#             depth=4,  # depth, paper recommended 6
#             heads=2,  # heads, paper recommends 8
#             attn_dropout=0.1,  # post-attention dropout
#             # ff_dropout=0.1  # feed forward dropout
#         )
#
#         optim = torch.optim.Adam(self.model.parameters(), lr=2.25e-3)
#
#         for _ in range(30):
#             x_categ = torch.tensor([[]])
#             clf = self.model(x_categ, xs_meta)
#
#             loss = torch.nn.functional.cross_entropy(clf, ys_meta.squeeze())
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#
#     def get_acc(self, xs_target, ys_target):
#         self.model.eval()
#         with torch.no_grad():
#             target_preds = self.model(self.null_categ, xs_target)
#         preds = torch.argmax(target_preds, dim=1)
#
#         return (preds == ys_target).numpy()
#
#     def __repr__(self):
#         return "FTTransformer"
#
#
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
                self.model = CatBoostClassifier(iterations=200, learning_rate=0.03, allow_const_label=True, verbose=False)
                # iterations=20, depth=4, learning_rate=0.5,
                #                             loss_function='Logloss', allow_const_label=True, verbose=False)

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
                # print(self.model.get_all_params())
                # exit(2)
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


class FLAT(Model):
    def __init__(self, load_no, save_ep=None):
        save_dir = f'{BASEDIR}/saves/save_{load_no}'
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')

        cfg = load_config(f'{save_dir}/config.toml')

        self.model = ModelHolder(cfg=cfg)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def get_accuracy(self, batch):
        xs_metas, ys_metas, xs_targets, ys_targets, _ = batch
        accs = []

        xs_metas, ys_metas, xs_targets, ys_targets = xs_metas[:max_batches + 1], ys_metas[:max_batches + 1], xs_targets[:max_batches + 1], ys_targets[
                                                                                                                                           :max_batches + 1]
        self.fit(xs_metas, ys_metas)
        a = self.get_acc(xs_targets, ys_targets)
        accs.append(a)

        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])

        return mean, std

    def fit(self, xs_meta, ys_meta):
        self.unique_ys_meta = np.unique(ys_meta)

        with torch.no_grad():
            self.pos_enc = self.model.forward_meta(xs_meta, ys_meta)

    def get_acc(self, xs_target, ys_target) -> np.array:

        unique_target = np.unique(ys_target)
        unique_labels = np.union1d(self.unique_ys_meta, unique_target)
        max_N_label = np.max(unique_labels) + 1

        with torch.no_grad():
            ys_pred_targ = self.model.forward_target(xs_target, self.pos_enc, max_N_label)
        predicted_labels = torch.argmax(ys_pred_targ, dim=1)

        return torch.eq(predicted_labels, ys_target.flatten()).numpy()

    def __repr__(self):
        return "FLAT"


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
    # [FLAT_MAML(num) for num in load_no] + \
    #  [
    #  BasicModel("LR"), # BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
    #  # TabnetModel(),
    #  # FTTrModel(),
    #  # STUNT(),
    #  ]

    test_results = get_results_by_dataset(test_datasets, models, N_meta=N_meta, N_target=N_target)

    process_results(test_results)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main(load_no=[-1], N_meta=5)

    # -1        58.34±0.34        NaN  64.27±0.33  64.18±0.33     -5.93            nan
