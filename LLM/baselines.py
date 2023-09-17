# Evaluate models on batches. Do the actual accuracy evaluation.

import os, toml, random
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, CatboostError
from tab_transformer_pytorch import FTTransformer
from xgboost import XGBClassifier

BASEDIR = '.'
max_batches = 40


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


class BasicModel(Model):
    def __init__(self, name):
        match name:
            case "LR":
                self.model = LogisticRegression(max_iter=1000)
            case "SVC":
                self.model = SVC(C=20, kernel="sigmoid", gamma='scale')
            case "KNN":
                self.model = KNN(n_neighbors=5, p=1, weights="distance")
            case "CatBoost":
                self.model = CatBoostClassifier(iterations=500, learning_rate=0.03, allow_const_label=True, verbose=False, auto_class_weights='Balanced')
            case "R_Forest":
                self.model = RandomForestClassifier(n_estimators=150, n_jobs=5)
            case "XGBoost":
                self.model = XGBClassifier(n_estimators=150, n_jobs=5)
            case _:
                raise Exception("Invalid model specified")

        self.name = name
        self.identical_batch = False

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten()
        xs_meta = xs_meta
        if ys_meta.min() == ys_meta.max():
            print("Catboost error")

            self.identical_batch = True
            self.pred_val = ys_meta[0]
        else:
            self.identical_batch = False

            try:
                self.model.fit(xs_meta, ys_meta)
            except CatboostError:
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
