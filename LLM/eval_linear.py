# Evaluate baseline models. Save results to file.
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import datetime

from modified_LR import LogRegBias, MonatoneLogReg
from baselines import BasicModel, OptimisedModel
from datasets import (Dataset, Adult, Bank, Blood, balanced_batches,
                      California, Diabetes, Heart, Jungle, Car, CreditG)


# Evaluate LR with biases
def LR_acc(data, size, bias: torch.Tensor = None, mask: torch.Tensor = None):
    X_train, _, _, _ = data[0]
    if mask is None:
        mask = torch.ones(X_train.shape[1])
    if bias is None:
        bias = torch.zeros(X_train.shape[1])
    bias.requires_grad = False
    mask.requires_grad = False

    lam = 0.5 / np.sqrt(size)

    accs, aucs = [], []
    for X_train, X_test, y_train, y_test in data[1:]:
        clf = LogRegBias(fit_intercept=True, lr=0.01, steps=100, lam=lam, bias=bias, mask=mask)
        clf.fit(X_train, y_train)
        acc, auc = clf.get_acc(X_test, y_test)
        accs.append(acc), aucs.append(auc)
        # print(acc, auc)

    std = np.std(aucs)
    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, aucs, std


def eval_ordering(ds, col_no, size, train_size, n_trials=10):
    ys = ds.num_data[:, -1]
    xs_ord = ds.get_ordered(col_no)

    ord_data = balanced_batches(xs_ord, ys, bs=train_size, num_batches=n_trials, seed=0)

    # Bias for LR
    bias, mask = ds.get_bias(col_no)
    bias, mask = torch.tensor(bias, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    mask = torch.ones_like(mask)

    acc, auc, std = LR_acc(ord_data, size=size, bias=bias, mask=mask)

    # Append results to file. Include time of current save to seperate saves.
    now = datetime.datetime.now()
    now_fmt = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(f'./results/lr_bias_results.txt', 'a') as f:
        f.write(f'\n{now_fmt}\n')
        print(f'{train_size} {auc:.3g}')
        f.write(f'{train_size} {acc:.3g}, {auc:.3g}, {std:.3g}\n')


def main():
    dl = Dataset(CreditG())
    cols = range(len(dl))

    print("Using columns:", dl.ds_prop.col_headers[cols])
    print()

    # List of models to evaluate
    for size in [4, 8, 16, 32, 64, 128, 256, 512]:
        eval_ordering(dl, cols, size=size, train_size=size, n_trials=120)


if __name__ == "__main__":
    main()
