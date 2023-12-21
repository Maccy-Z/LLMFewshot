# Evaluate baseline models. Save results to file.
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import datetime, time

from optim_baseline import BasicModel, OptimisedModel
from datasets import (Dataset, Adult, Bank, Blood, balanced_batches,
                      California, Diabetes, Heart, Jungle, Car, CreditG)


# Evaluate model and fit hyperparameters
def optim_acc(data, model):
    clf = OptimisedModel(model)

    accs, aucs = [], []
    for i, (X_train, X_test, y_train, y_test) in enumerate(data):

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
    std = np.std(aucs)
    accs, aucs = np.mean(accs), np.mean(aucs)

    return accs, aucs, std


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
    # TODO: Enter dataset here.
    dl = Dataset(CreditG())
    cols = range(len(dl))

    print("Using columns:", dl.ds_prop.col_headers[cols])
    print()

    # List of models to evaluate
    model_list = [
        # ("LightGBM", ["raw", "order", "onehot"]),
        # ("LR", ['raw', "order", "onehot"]),
        # ("XGBoost", ["raw", "order", "onehot"]),
        ("TabPFN", ["raw", "order", "onehot"]),
    ]

    for size in [4, 8, 16, 32, 64, 128, 256, 512]:
        eval_ordering(model_list, dl, cols, train_size=size, n_trials=20)


if __name__ == "__main__":
    main()
