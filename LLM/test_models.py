import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch

from modified_LR import LogRegBias, MonatoneLogReg
from baselines import BasicModel
from datasets import Dataset, Adult, Bank


def monat_acc(data):
    X_train, X_test, y_train, y_test = data

    clf = MonatoneLogReg(steps=200, lam=0.05, lr=0.02)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    clf.plot_net()
    return acc, auc


def base_acc(data, model):
    X_train, X_test, y_train, y_test = data

    clf = BasicModel(model)
    clf.fit(X_train, y_train)
    acc = clf.get_acc(X_test, y_test).mean()

    y_prob = clf.predict_proba(X_test)
    y_prob = y_prob[:, -1]
    auc = roc_auc_score(y_test, y_prob)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def LR_acc(data, lam=0.01, bias: torch.Tensor = None, mask: torch.Tensor = None):
    X_train, X_test, y_train, y_test = data
    if mask is None:
        mask = torch.ones(X_train.shape[1])
    if bias is None:
        bias = torch.zeros(X_train.shape[1])
    bias.requires_grad = False
    mask.requires_grad = False

    clf = LogRegBias(fit_intercept=True, lam=lam, bias=bias, mask=mask)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)
    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def eval_ordering(ds, col_no, train_size, seed):
    ys = ds.num_data[:, -1]
    accs, aucs = [], []

    xs_raw = ds.get_base(col_no)
    xs_ord = ds.get_ordered(col_no)
    xs_one = ds.get_onehot(col_no)

    raw = train_test_split(xs_raw, ys, train_size=train_size, random_state=seed, stratify=ys)
    ord = train_test_split(xs_ord, ys, train_size=train_size, random_state=seed, stratify=ys)
    one = train_test_split(xs_one, ys, train_size=train_size, random_state=seed, stratify=ys)

    # print("Baseline")
    # a, auc = LR_acc(raw)
    # accs.append(a), aucs.append(auc)
    #
    # print("One hot")
    # a, auc = LR_acc(one)
    # accs.append(a), aucs.append(auc)
    #
    # print("Ordered")
    # a, auc = LR_acc(ord)
    # accs.append(a), aucs.append(auc)
    #
    # print("Biased")
    # bias, mask = ds.get_bias(col_no)
    # # mask = np.ones_like(mask)
    # bias, mask = torch.tensor(bias, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    # mask = torch.ones_like(mask)
    # a, auc = LR_acc(ord, lam=0.01, bias=bias)
    # accs.append(a), aucs.append(auc)
    #
    # print("CatBoost base")
    # a, auc = base_acc(raw, "CatBoost")
    # accs.append(a), aucs.append(auc)
    #
    # print("CatBoost ordered")
    # a, auc = base_acc(ord, "CatBoost")
    # accs.append(a), aucs.append(auc)
    #
    # print("CatBoost one-hot")
    # a, auc = base_acc(one, "CatBoost")
    # accs.append(a), aucs.append(auc)

    print("Monat ordered")
    a, auc = monat_acc(ord)
    accs.append(a), aucs.append(auc)

    return accs, aucs


def main():

    ds = Bank()
    dl = Dataset(ds)
    cols = range(len(dl))
    print("Using columns:", ds.col_headers[cols])

    accs, aucs = [], []
    for s in range(10):
        print()
        acc, auc = eval_ordering(dl, cols, train_size=512, seed=s)
        accs.append(acc), aucs.append(auc)

    accs, aucs = np.array(accs), np.array(aucs)
    mean, aucs = np.mean(accs, axis=0), np.mean(aucs, axis=0)
    print()
    [print(f'acc = {m:.3g}, auc = {a:.3g}') for m, a in zip(mean, aucs)]


if __name__ == "__main__":
    main()

# 0.7670200563987198
# 0.800307975032477
# 0.7963904819238934
