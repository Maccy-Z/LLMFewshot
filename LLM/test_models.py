import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch

from modified_LR import TorchLogReg
from baselines import BasicModel
from datasets import Adult, Dataset


def base_acc(xs, ys, train_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=seed, stratify=ys)

    clf = BasicModel("CatBoost")
    clf.fit(X_train, y_train)
    acc = clf.get_acc(X_test, y_test).mean()

    y_prob = clf.predict_proba(X_test)
    y_prob = y_prob[:, -1]
    auc = roc_auc_score(y_test, y_prob)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def LR_acc(xs, ys, train_size, seed, lam=0., bias: torch.Tensor = None, mask: torch.Tensor = None):
    if mask is None:
        mask = torch.ones(xs.shape[1])
    if bias is None:
        bias = torch.zeros(xs.shape[1])

    bias.requires_grad = False
    mask.requires_grad = False

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=seed, stratify=ys)

    clf = TorchLogReg(fit_intercept=True, lam=lam, bias=bias, mask=mask)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)
    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    return acc, auc


def eval_ordering(ds, col_no, train_size, seed):
    ys = ds.num_data[:, -1]
    accs, aucs = [], []

    xs_raw = ds.get_base(col_no)
    xs_ord = ds.get_ordered(col_no)

    print("Baseline")
    a, auc = LR_acc(xs_raw, ys, train_size, seed)
    accs.append(a), aucs.append(auc)

    print("Ordered")
    a, auc = LR_acc(xs_ord, ys, train_size, seed)
    accs.append(a), aucs.append(auc)

    print("Biased")
    bias, mask = ds.get_bias(col_no)
    # mask = np.ones_like(mask)
    bias, mask = torch.tensor(bias, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    a, auc = LR_acc(xs_ord, ys, train_size, seed, lam=0.01, bias=bias, mask=mask)
    accs.append(a), aucs.append(auc)

    print("SVC base")
    a, auc = base_acc(xs_raw, ys, train_size, seed)
    accs.append(a), aucs.append(auc)

    print("SVC ordered")
    a, auc = base_acc(xs_ord, ys, train_size, seed)
    accs.append(a), aucs.append(auc)

    return accs, aucs


if __name__ == "__main__":

    ds = Adult()
    dl = Dataset(ds)
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print("Using columns:", ds.col_headers[cols])

    accs, aucs = [], []
    for s in range(5):
        print()
        acc, auc = eval_ordering(dl, cols, train_size=10, seed=s)
        accs.append(acc), aucs.append(auc)

    accs, aucs = np.array(accs), np.array(aucs)
    mean, aucs = np.mean(accs, axis=0), np.mean(aucs, axis=0)
    print()
    [print(f'acc = {m:.3g}, auc = {a:.3g}') for m, a in zip(mean, aucs)]

# 0.7670200563987198
# 0.800307975032477
# 0.7963904819238934
