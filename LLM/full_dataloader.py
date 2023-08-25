import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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
    col_headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                   "native-country"]
    col_dtypes = [float, str, float, str, float, str,
                  str, str, str, str, float, float, float,
                  str, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {1: ['Self-emp-inc', 'Federal-gov', 'Self-emp-not-inc', 'Private', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
                      3: ["Doctorate", "Prof-school", "Masters", "Bachelors", "Assoc-acdm", "Assoc-voc", "Some-college",
                          "HS-grad", "12th", "11th", "10th", "9th", "7th-8th", "5th-6th", "1st-4th", "Preschool"],
                      5: ['Married-civ-spouse', 'Married-AF-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'],
                      7: ['Husband', 'Wife', 'Not-in-family', 'Unmarried', 'Own-child', 'Other-relative'],
                      }

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

    def str_to_order_int(self, col_no):
        print(f'Mapping column number {col_no}, {self.col_to_head[col_no]}')

        map = {s: i for i, s in enumerate(self.ordered_labels[col_no])}
        ordered_data = [map[s] for s in self.data[:, col_no]]
        ordered_data = np.array(ordered_data, dtype=float)

        return ordered_data


def LR_acc(xs, ys):
    xs = xs.reshape(-1, 1)
    mean, std = np.mean(xs), np.std(xs)
    xs = (xs - mean) / (std + 1e-6)
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=0)

    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f'beta = {clf.coef_[0]}, c = {clf.intercept_[0]}')


def eval_ordering(col_no):
    ds = AdultDataset()

    ys = ds.num_data[:, -1]

    print()
    print("Baseline")
    xs_raw = ds.num_data[:, col_no]
    LR_acc(xs_raw, ys)

    print()
    print("Ordered")
    xs_ord = ds.str_to_order_int(col_no)
    LR_acc(xs_ord, ys)

# def main():
#     work_ordered = ['Self-emp-inc', 'Federal-gov', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Private', 'Without-pay', 'Never-worked', '?', ]
#     data, num_data = data_process()
#
#     ys = num_data[:, -1]
#
#     for work_type in work_ordered:
#         print()
#         print(work_type)
#         want_rows = (data[:, 1] == work_type)
#         print(np.mean(ys[want_rows]))


if __name__ == "__main__":
    eval_ordering(1)
