# Evaluate baseline models. Save results to file.
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import datetime

from modified_LR import LogRegBias, MonatoneLogReg
from baselines import BasicModel, OptimisedModel
from datasets import Dataset, Adult, Bank, balanced_batches


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

    return accs, aucs, np.std(aucs)


def eval_ordering(model_list, ds, col_no, train_size, n_trials=10):
    ys = ds.num_data[:, -1]

    xs_raw = ds.get_base(col_no)
    xs_ord = ds.get_ordered(col_no)
    xs_one = ds.get_onehot(col_no)

    raw_data = balanced_batches(xs_raw, ys, bs=train_size, num_batches=n_trials, seed=0)
    ord_data = balanced_batches(xs_ord, ys, bs=train_size, num_batches=n_trials, seed=0)
    onehot_data = balanced_batches(xs_one, ys, bs=train_size, num_batches=n_trials, seed=0)

    for model_type, eval_types in model_list:
        results = []
        if "raw" in eval_types:
            a, auc, std = optim_acc(raw_data, model_type)
            results.append(["raw", a, auc, std])
        if "order" in eval_types:
            a, auc, std = optim_acc(ord_data, model_type)
            results.append(["raw", a, auc, std])
        if "onehot" in eval_types:
            a, auc, std = optim_acc(onehot_data, model_type)
            results.append(["raw", a, auc, std])

        print(f'{model_type}')
        for r in results:
            print(f'{r[0]}: accuracy: {r[1]:.3g}, auc: {r[2]:.3g}')
        print()

        # Append results to file. Include time of current save to seperate saves.
        now = datetime.datetime.now()
        now_fmt = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(f'./results/{model_type}_results.txt', 'a') as f:
            f.write(f'\n{now_fmt}\n')
            for r in results:
                print(f'{train_size} {r[2]:.3g}')

                f.write(f'{r[0]} {train_size} {r[1]:.3g}, {r[2]:.3g}, {r[3]:.3g}\n')


def main():
    dl = Dataset(Bank())
    cols = range(len(dl))

    print("Using columns:", dl.ds_prop.col_headers[cols])
    print()

    # List of models to evaluate
    model_list = [
        ("LightGBM", ["raw", "order", "onehot"]),
        ("LR", ['raw', "order", "onehot"]),
        ("XGBoost", ["raw", "order", "onehot"]),
    ]

    for size in [4, 8, 16]:
        eval_ordering(model_list, dl, cols, train_size=size, n_trials=120)


if __name__ == "__main__":
    main()
