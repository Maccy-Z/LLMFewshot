import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch

from modified_LR import LogRegBias, MonatoneLogReg
from baselines import BasicModel, OptimisedModel
from datasets import Dataset, Adult, Bank


def monat_acc(data):
    X_train, X_test, y_train, y_test = data

    clf = MonatoneLogReg(steps=200, lam=0.5, lr=0.02)
    clf.fit(X_train, y_train)
    acc, auc = clf.get_acc(X_test, y_test)

    print(f'Accuracy: {acc:.3g}, {auc = :.3g}')

    clf.plot_net()
    return acc, auc


# Evaluate LR with biases
def LR_acc(data, bias: torch.Tensor = None, mask: torch.Tensor = None):
    X_train, _, _, _ = data[0]
    if mask is None:
        mask = torch.ones(X_train.shape[1])
    if bias is None:
        bias = torch.zeros(X_train.shape[1])
    bias.requires_grad = False
    mask.requires_grad = False

    accs, aucs = [], []
    for X_train, X_test, y_train, y_test in data[1:]:
        clf = LogRegBias(fit_intercept=True, lr=0.01, steps=100, lam=0.1, bias=bias, mask=mask)
        clf.fit(X_train, y_train)
        acc, auc = clf.get_acc(X_test, y_test)
        accs.append(acc), aucs.append(auc)
        # print(acc, auc)

    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, aucs


# Evaluate model with prefit paramters
def base_acc(data, model):
    clf = BasicModel(model)
    # Evaluate optimal model on test set
    accs, aucs = [], []
    for X_train, X_test, y_train, y_test in data[1:]:
        clf.fit(X_train, y_train)
        acc = clf.get_acc(X_test, y_test).mean()

        y_prob = clf.predict_proba(X_test)
        y_prob = y_prob[:, -1]
        auc = roc_auc_score(y_test, y_prob)

        accs.append(acc), aucs.append(auc)

    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, aucs


# Evaluate model and fit hyperparameters
def optim_acc(data, model):
    clf = OptimisedModel(model)

    accs, aucs = [], []
    for i, (X_train, X_test, y_train, y_test) in enumerate(data):
        # Rerun paramter optimisation every few trials to avoid overfitting to single batch
        if i % 20 == 0:
            clf.fit_params(X_train, y_train)
            continue

        clf.fit(X_train, y_train)
        acc = clf.get_acc(X_test, y_test).mean()
        y_prob = clf.predict_proba(X_test)[:, -1]
        auc = roc_auc_score(y_test, y_prob)

        accs.append(acc), aucs.append(auc)

    accs, aucs = np.array(accs), np.array(aucs)
    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, aucs


def eval_ordering(model_list, ds, col_no, train_size, n_trials=10):
    ys = ds.num_data[:, -1]

    xs_raw = ds.get_base(col_no)
    xs_ord = ds.get_ordered(col_no)
    xs_one = ds.get_onehot(col_no)

    raw_data, ord_data, onehot_data = [], [], []
    for s in range(1, n_trials + 1):
        raw = train_test_split(xs_raw, ys, train_size=train_size, random_state=s, stratify=ys)
        order = train_test_split(xs_ord, ys, train_size=train_size, random_state=s, stratify=ys)
        one = train_test_split(xs_one, ys, train_size=train_size, random_state=s, stratify=ys)

        raw_data.append(raw), ord_data.append(order), onehot_data.append(one)

    for model_type, eval_types in model_list:
        results = []
        if "raw" in eval_types:
            a, auc = optim_acc(raw_data, model_type)
            results.append(["raw", a, auc])
        if "order" in eval_types:
            a, auc = optim_acc(ord_data, model_type)
            results.append(["ord", a, auc])
        if "onehot" in eval_types:
            a, auc = optim_acc(onehot_data, model_type)
            results.append(["onehot", a, auc])

        print(f'{model_type}')
        for r in results:
            print(f'{r[0]}: accuracy: {r[1]:.3g}, auc: {r[2]:.3g}')
        print()

        with open(f'./results/{model_type}_results.txt', 'a') as f:
            for r in results:
                print(f'{train_size} {r[2]:.3g}')

                f.write(f'{r[0]} {train_size} {r[2]:.3g}\n')


def main():
    dl = Dataset(Bank())
    cols = range(len(dl))

    print("Using columns:", dl.ds_prop.col_headers[cols])
    print()

    # List of models to evaluate
    model_list = [("LightGBM", ["raw", "order", "onehot"]),
                  # ("LR", ["order", "onehot"]),
                  ("XGBoost", ["raw", "order", "onehot"]),
                  ]

    eval_ordering(model_list, dl, cols, train_size=512, n_trials=100)


if __name__ == "__main__":
    main()
