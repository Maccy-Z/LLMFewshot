# Evaluate models on batches. Do the actual accuracy evaluation.
import numpy as np
import sklearn.base
from scipy import stats
from abc import ABC, abstractmethod
from collections import Counter
from utils import c_print

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier, CatboostError
from xgboost import XGBClassifier
import lightgbm as gb
from tabpfn import TabPFNClassifier

BASEDIR = '.'
max_batches = 1000


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

class OptimisedModel(Model):
    param_grid: dict
    best_params: dict
    model: sklearn.base.BaseEstimator | CatBoostClassifier | TabPFNClassifier

    def __init__(self, name):
        self.name = name
        self.identical_batch = False
        self.best_params = {}

    # Sets model, using best hyperparameters if set. If not, uses default hyperparameters.
    def set_model(self):
        match self.name:
            case "LR":
                self.model = LogisticRegression(**self.best_params, max_iter=200, n_jobs=1)
                self.param_grid = {"C": [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
            case "SVC":
                self.model = SVC(**self.best_params)
            case "KNN":
                self.model = KNN(**self.best_params)
            case "CatBoost":
                self.model = CatBoostClassifier(**self.best_params, verbose=False, auto_class_weights='Balanced', thread_count=1)
                self.param_grid = {"iterations": [10, 100, 250, 500],
                                   "learning_rate": [0.005, 0.01, 0.03],
                                   "depth": [2, 4, 6, 8],
                                   "l2_leaf_reg": [1e-8, 1e-6, 1e-4],
                                   }
            case "R_Forest":
                self.model = RandomForestClassifier(**self.best_params)
            case "XGBoost":
                self.model = XGBClassifier(**self.best_params, objective='binary:logistic', nthread=1)
                self.param_grid = {"max_depth": [2, 4, 6, 8, 10, 12],
                                   "reg_alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                                   "reg_lambda": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                                   "eta": [0.01, 0.03, 0.1, 0.3]}
            case "LightGBM":
                self.model = gb.LGBMClassifier(**self.best_params, verbosity=-1, num_threads=1)
                self.param_grid = {"num_leaves": [2, 4, 8, 16, 32, 64],
                                   "lambda_l1": [1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
                                   "lambda_l2": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1],
                                   "learning_rate": [0.01, 0.03, 0.1]}
            case "TabPFN":
                self.model = TabPFNClassifier(device="cpu", batch_size_inference=32)
                self.param_grid = {}
            case _:
                raise Exception("Invalid model specified")

    def fit_params(self, xs_val, ys_val):
        ys_val = ys_val.flatten()
        if ys_val.min() == ys_val.max():
            return

        # Find optimal paramters
        self.best_params = {}
        self.set_model()
        folds = min(Counter(ys_val).values()) if min(Counter(ys_val).values()) < 4 else 4

        if folds > 1:
            inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        else:
            folds = 2
            c_print(f"Warning: Increased folds from {folds} to 2 (even though not enough labels) and use simple KFold.", color="red")

            inner_cv = KFold(n_splits=folds, shuffle=True, random_state=0)

        grid_search = GridSearchCV(self.model, self.param_grid, cv=inner_cv, scoring='accuracy', verbose=0, n_jobs=8)

        grid_search.fit(xs_val, ys_val)

        # Make model with optimal parameters
        self.best_params = grid_search.best_params_

        c_print(f'{self}, hyperparams: {self.best_params}', color="yellow")

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten()
        if ys_meta.min() == ys_meta.max():
            print("Identical elements in batch")

            self.identical_batch = True
            self.pred_val = ys_meta[0]
        else:
            self.identical_batch = False
            # Allow calling fit() multiple times by resetting model each time.
            self.set_model()
            try:
                self.model.fit(xs_meta, ys_meta)
            except CatboostError as e:
                # Catboost fails if every input element is the same
                self.identical_batch = True
                mode = stats.mode(ys_meta, keepdims=False)[0]
                self.pred_val = mode

    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            predictions = self.model.predict(xs_target)

        return np.array(predictions).flatten() == np.array(ys_target)

    def predict_proba(self, xs_target):
        if self.identical_batch:
            return np.ones([xs_target.shape[0], 2]) * 0.5
        else:
            return self.model.predict_proba(xs_target)

    def __repr__(self):
        return self.name
