import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

from modified_LR import TorchLogReg
from Fewshot import config

def read_csv_to_2d_array(filename):
    """Read data from a CSV file into a 2D array."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cleaned_row = [entry.replace(' ', '') for entry in row]
            data.append(cleaned_row)
    return data


def map_strings_to_int(xs):
    """Convert a list of strings to a list of integers based on the unique and sorted order of the strings."""

    # Identify unique values and sort them
    unique_sorted_values = sorted(set(xs))

    # print(unique_sorted_values)
    # Create a dictionary that maps each unique string value to a unique integer
    value_to_int = {value: i for i, value in enumerate(unique_sorted_values)}

    # Convert the original list into a list of integers using the dictionary
    mapped_data = [value_to_int[item] for item in xs]

    return np.array(mapped_data, dtype=float)


class AdultDataset:
    col_headers = np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                            "native-country"])
    col_dtypes = [float, str, float, str, float, str,
                  str, str, str, str, float, float, float,
                  str, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {1: ['Self-emp-inc', 'Federal-gov', 'Self-emp-not-inc', 'Private', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                      3: ["Doctorate", "Prof-school", "Masters", "Bachelors", "Assoc-acdm", "Assoc-voc", "Some-college",
                          "HS-grad", "12th", "11th", "10th", "9th", "7th-8th", "5th-6th", "1st-4th", "Preschool"],
                      5: ['Married-civ-spouse', 'Married-AF-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'],
                      6: ['Exec-managerial', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Craft-repair', 'Adm-clerical', '?', 'Machine-op-inspct',
                          'Farming-fishing', 'Handlers-cleaners', 'Other-service', 'Priv-house-serv', 'Armed-Forces'],
                      7: ['Husband', 'Wife', 'Not-in-family', 'Unmarried', 'Own-child', 'Other-relative'],
                      }
    correl_coef = {0: 1, 1: -1, 2: 0, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 13: 0}

    def __init__(self):
        filename = "./adult/adult.data"
        data_2d_array = read_csv_to_2d_array(filename)[:-1]
        data = np.array(data_2d_array)

        col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}

        # Map all data to float
        numerical_data = []
        for j in range(data.shape[1]):
            col = data[:, j]
            if col_to_dtype[j] == str:
                float_col = map_strings_to_int(col)
            else:
                float_col = col.astype(float)
            numerical_data.append(float_col)

        numerical_data = np.stack(numerical_data).T

        self.data = data
        self.num_data = numerical_data
        self.col_to_head = col_to_head
        self.col_to_dtype = col_to_dtype

        self.ordered_data = numerical_data.copy()
        for c in self.ordered_labels.keys():
            self.ordered_data[:, c] = self.str_to_order_int(c)

    def str_to_order_int(self, col_no):

        print(f'Mapping column number {col_no}, {self.col_to_head[col_no]}')

        order_map = {s: i for i, s in enumerate(self.ordered_labels[col_no])}
        ordered_data = [order_map[s] for s in self.data[:, col_no]]
        ordered_data = np.array(ordered_data, dtype=float)

        return ordered_data

    def get_ordered(self):
        return self.ordered_data

    def get_base(self):
        return self.num_data

    def get_bias(self):
        return self.correl_coef


def LR_acc(xs, ys, train_size, seed, lam=0., bias: float | torch.Tensor = 0.):
    if len(xs.shape) == 1:
        xs = xs[:, np.newaxis]
    mean, std = np.mean(xs, axis=0), np.std(xs, axis=0)

    xs = (xs - mean) / (std + 1e-3)
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=seed)

    # clf = LogisticRegression(class_weight="balanced")
    clf = TorchLogReg(fit_intercept=True, lam=lam, bias=bias)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pos_acc = tp / (tp + fn)
    neg_acc = tn / (tn + fp)

    print(f'{pos_acc = :.3g}, {neg_acc = :.3g}')

    # print(f'beta = {clf.coef_[0]}, c = {clf.intercept_[0]}'

    return accuracy, pos_acc, neg_acc


def eval_ordering(ds, col_no, train_size, seed):
    ys = ds.num_data[:, -1]
    accs = []

    print("Baseline")
    xs_raw = ds.num_data[:, col_no]
    a, _, _ = LR_acc(xs_raw, ys, train_size, seed)
    accs.append(a)

    print("Ordered")
    xs_ord = ds.ordered_data[:, col_no]
    a, _, _ = LR_acc(xs_ord, ys, train_size, seed)
    accs.append(a)

    print("Biased")
    bias = [ds.correl_coef[c] for c in col_no]
    # print("Bias term:", bias)
    a, _, _ = LR_acc(xs_ord, ys, train_size, seed, lam=0.025, bias=torch.tensor(bias, dtype=torch.float32))
    accs.append(a)

    return accs


if __name__ == "__main__":

    accs = []
    ds = AdultDataset()
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print("Using columns:", ds.col_headers[cols])

    for s in range(25):
        print()
        a = eval_ordering(ds, cols, train_size=1000, seed=s)
        accs.append(a)

    accs = np.array(accs)
    mean = np.mean(accs, axis=0)
    [print(m) for m in mean]
