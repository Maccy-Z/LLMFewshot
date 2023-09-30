from baselines import OptimisedModel

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datasets import Adult, Dataset, Bank
from sklearn.metrics import roc_auc_score

ds = Dataset(Adult())

xs_raw = ds.get_base(range(len(ds)))
ys = ds.num_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(xs_raw, ys, train_size=512, random_state=0, stratify=ys)

model = OptimisedModel("CatBoost")
print("Searching hyperparamters...")
model.fit_params(X_train, y_train)

print("Fitting mode")
model.fit(X_train, y_train)
print("Getting accuracy")
acc = model.get_acc(X_test, y_test).mean()
y_prob = model.predict_proba(X_test)[:, -1]
auc = roc_auc_score(y_test, y_prob)


print(f"Accuracy using best hyperparameters: {acc:.2f}")
print(f'auroc using best hyperparameters: {auc:.2f}')
# Best hyperparameters: {'eta': 0.3, 'max_depth': 6, 'reg_alpha': 0.001, 'reg_lambda': 0.001}
# Best hyperparameters: {'eta': 0.1, 'max_depth': 6, 'reg_alpha': 0.001, 'reg_lambda': 0.0001}
# print("Best hyperparameters:", grid_search.best_params_)
