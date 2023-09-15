import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch

from modified_LR import LogRegBias, MonatoneLogReg
from baselines import BasicModel
from datasets import Dataset, Adult, Bank


def monat_acc(data):
    X_train, X_test, y_train, y_test = data

    clf = MonatoneLogReg(steps=200)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def base_acc(data):
    X_train, X_test, y_train, y_test = data

    clf = BasicModel("CatBoost")
    clf.fit(X_train, y_train)
    acc = clf.get_acc(X_test, y_test).mean()

    y_prob = clf.predict_proba(X_test)
    y_prob = y_prob[:, -1]
    auc = roc_auc_score(y_test, y_prob)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def LR_acc(data, lam=0., bias: torch.Tensor = None, mask: torch.Tensor = None):
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

    raw = X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(xs_raw, ys, train_size=train_size, random_state=seed, stratify=ys)
    ord = X_ord_train, X_ord_test, y_ord_train, y_ord_test = train_test_split(xs_ord, ys, train_size=train_size, random_state=seed, stratify=ys)

    print("Baseline")
    a, auc = LR_acc(raw)
    accs.append(a), aucs.append(auc)

    print("Ordered")
    a, auc = LR_acc(ord)
    accs.append(a), aucs.append(auc)

    print("Biased")
    bias, mask = ds.get_bias(col_no)
    # mask = np.ones_like(mask)
    bias, mask = torch.tensor(bias, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    a, auc = LR_acc(ord, lam=0.01, bias=bias, mask=mask)
    accs.append(a), aucs.append(auc)

    print("SVC base")
    a, auc = base_acc(raw)
    accs.append(a), aucs.append(auc)

    print("Monat ordered")
    a, auc = monat_acc(ord)
    accs.append(a), aucs.append(auc)

    return accs, aucs


if __name__ == "__main__":

    ds = Adult()
    dl = Dataset(ds)
    cols = range(14)
    print("Using columns:", ds.col_headers[cols])

    accs, aucs = [], []
    for s in range(20):
        print()
        acc, auc = eval_ordering(dl, cols, train_size=100, seed=s)
        accs.append(acc), aucs.append(auc)

    accs, aucs = np.array(accs), np.array(aucs)
    mean, aucs = np.mean(accs, axis=0), np.mean(aucs, axis=0)
    print()
    [print(f'acc = {m:.3g}, auc = {a:.3g}') for m, a in zip(mean, aucs)]

# 0.7670200563987198
# 0.800307975032477
# 0.7963904819238934
