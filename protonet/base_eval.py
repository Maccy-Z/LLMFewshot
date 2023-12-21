# Evaluate baseline models. Save results to file.
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import itertools

from base_optim import BasicModel, OptimisedModel
from dataloader import SplitDataloader
from config import Config
from utils import c_print


# Evaluate model and fit hyperparameters
def optim_acc(ds, model, cfg):
    clf = OptimisedModel(model)

    batch = next(iter(ds))
    batch = batch[:-1]  # Ignore extra batch data

    accs, aucs = [], []
    for i, (X_train, y_train, X_test, y_test) in enumerate(zip(*batch)):
        # Convert from torch to numpy
        X_train, y_train, X_test, y_test = X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy()
        # Rerun parameter optimisation every few trials to avoid overfitting to single batch
        if i % 20 == 0:
            clf.fit_params(X_train, y_train)
            continue

        clf.fit(X_train, y_train)

        if len(set(y_test)) == 2:
            # Binary classification
            y_prob = clf.predict_proba(X_test)
            preds = np.argmax(y_prob, axis=1)

            acc = np.mean(preds == y_test)
            auc = roc_auc_score(y_test, y_prob[:, -1])
        else:
            # Multiclass classification
            y_prob = clf.predict_proba(X_test)  # [:, -1]
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

            preds = np.argmax(y_prob, axis=1)
            acc = np.mean(preds == y_test)

        accs.append(acc), aucs.append(auc)

    accs, aucs = np.array(accs), np.array(aucs)
    std = np.std(aucs) / np.sqrt(len(aucs))
    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, std, aucs


def eval_ordering(model_list, data, cfg):
    for model_type in model_list:
        acc, std, auc = optim_acc(data, model_type, cfg)

        print(f'N_meta = {cfg.N_meta}, accuracy: {acc:.3g} +- {std:.3g}')

        # Append results to file. Include time of current save to seperate saves.
        # now = datetime.datetime.now()
        # now_fmt = now.strftime("%d/%m/%Y %H:%M:%S")
        # with open(f'./results/{model_type}_results.txt', 'a') as f:
        #     f.write(f'\n{now_fmt}\n')
        #     for r in results:
        #         print(f'{train_size} {r[2]:.3g}')
        #
        #         f.write(f'{r[0]} {train_size} {r[1]:.3g}, {r[2]:.3g}, {r[3]:.3g}\n')


def main():
    # TODO: Enter dataset here.

    print()

    # List of models to evaluate
    model_list = ["CatBoost", ]

    for N_meta in [128, 256, 512]:
        cfg = Config()

        cfg.N_meta = N_meta
        cfg.bs = 101
        ds = SplitDataloader(cfg, dataset='adult')
        eval_ordering(model_list, ds, cfg)


if __name__ == "__main__":
    main()
