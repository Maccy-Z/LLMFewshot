import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datasets import Adult, Dataset, Bank
from sklearn.metrics import roc_auc_score

ds = Dataset(Adult())

xs_raw = ds.get_base(range(len(ds)))
ys = ds.num_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(xs_raw, ys, train_size=32, random_state=0, stratify=ys)

clf = xgb.XGBClassifier(objective='binary:logistic')
param_grid = {"max_depth": [4, 6, 8, 10, 12],
              "reg_alpha": [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
              "reg_lambda": [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
              "eta": [0.01, 0.03, 0.1, 0.3]}

grid_search = GridSearchCV(clf, param_grid, cv=4, scoring='accuracy', verbose=3, n_jobs=7)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

y_pred = best_clf.predict_proba(X_test)[:, -1]

auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy using best hyperparameters: {accuracy:.2f}")
print(f'auroc using best hyperparameters: {auc:.2f}')
# Best hyperparameters: {'eta': 0.3, 'max_depth': 6, 'reg_alpha': 0.001, 'reg_lambda': 0.001}
# Best hyperparameters: {'eta': 0.1, 'max_depth': 6, 'reg_alpha': 0.001, 'reg_lambda': 0.0001}
print("Best hyperparameters:", grid_search.best_params_)
