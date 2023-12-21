# Plot individual predictions
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import datetime
from multiprocessing import Pool
from matplotlib import pyplot as plt

from modified_LR import LogRegBias, MonatoneLogReg
from datasets import (Dataset, Adult, Bank, Blood, balanced_batches,
                      California, Diabetes, Heart, Jungle, Car, map_strings_to_int)


def monat_single(data, lam, bias, unnorm):
    X_train, X_test, y_train, y_test = data
    X_unnorm, _, _, _ = unnorm
    clf = MonatoneLogReg(steps=150, lam=lam, lr=0.02, bias=bias)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)

    GET_COL = 7
    xs, preds = clf.plot_net(GET_COL)

    xs_col = X_unnorm[:, GET_COL]
    # xs_unique = list(set(xs_col))

    # Convert to continuous for discrete
    # xs_col = map_strings_to_int(xs_col)
    # xs_col = [int(x) for x in xs_col]

    xs_col = sorted(list(set(xs_col)))

    # Plot predictions
    plt.figure(figsize=(4, 4))
    plt.plot(xs_col, preds)

    # xticks = np.arange(0, len(xs_col), 1)
    # plt.xticks(xticks, xticks, fontsize=12)
    # yticks = np.arange(0.5, -1.5, 0.5)
    # plt.yticks(yticks, labels=yticks, fontsize=12)

    # plt.xlabel("Chest Pain Type", fontsize=13)
    plt.ylabel("Activation magnitude", fontsize=13)

    plt.tight_layout()
    plt.show()

    print()
    print()
    print(preds.tolist(), xs_col)
    # print()

    return acc, auc


def monat_acc_mp(data, size, unnorm_data, bias: torch.Tensor = None):
    X_train, _, _, _ = data[0]
    if bias is None:
        bias = torch.zeros(X_train.shape[1])
    bias.requires_grad = False
    lam = 0.1 / np.sqrt(size)

    args = [(batch, lam, bias, unnorm) for batch, unnorm in zip(data, unnorm_data)]
    with Pool(8) as p:
        vals = p.starmap(monat_single, args)

    vals = np.array(vals)
    accs, aucs = vals[:, 0], vals[:, 1]

    std = np.std(aucs)
    accs, aucs = np.mean(accs), np.mean(aucs)
    print(f'Acc: {accs:.3g}, {aucs = :.3g}')
    return accs, aucs, std


def monat_acc(data, size, bias: torch.Tensor = None, mask: torch.Tensor = None):
    X_train, _, _, _ = data[0]
    if mask is None:
        mask = torch.ones(X_train.shape[1])
    if bias is None:
        bias = torch.zeros(X_train.shape[1])
    bias.requires_grad = False
    mask.requires_grad = False
    lam = 0.5 / np.sqrt(size)

    accs, aucs = [], []
    for batch in data[1:]:
        acc, auc = monat_single(batch, lam, bias, mask)
        accs.append(acc), aucs.append(auc)

    std = np.std(aucs)
    accs, aucs = np.mean(accs), np.mean(aucs)
    print(f'Accuracy: {accs:.3g}, {aucs = :.3g}')
    return accs, aucs, std


def eval_ordering(ds, col_no, size, train_size, n_trials):
    ys = ds.num_data[:, -1]
    xs_ord = ds.get_ordered(col_no)
    xs_unnorm = ds.get_unnormaliesd(col_no)

    ord_data = balanced_batches(xs_ord, ys, bs=train_size, num_batches=n_trials, seed=0)
    unnorm_data = balanced_batches(xs_unnorm, ys, bs=train_size, num_batches=n_trials, seed=0)

    # Bias for LR
    bias, mask = ds.get_bias(col_no)
    bias = torch.tensor(bias, dtype=torch.float32)

    acc, auc, std = monat_acc_mp(ord_data, size=size, bias=bias, unnorm_data=unnorm_data)

    # Append results to file. Include time of current save to separate saves.
    now = datetime.datetime.now()
    now_fmt = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(f'./results/lr_monat_results.txt', 'a') as f:
        f.write(f'\n{now_fmt}\n')
        print(f'{train_size} {auc:.3g}')
        f.write(f'{train_size} {acc:.3g}, {auc:.3g}, {std:.3g}\n')


def main():
    dl = Dataset(California())
    cols = range(len(dl))

    print("Using columns:", dl.ds_prop.col_headers[cols])
    print()

    # List of models to evaluate
    for size in [1024]:
        eval_ordering(dl, cols, size=size, train_size=size, n_trials=8)


if __name__ == "__main__":
    main()
