import csv
import numpy as np
from sklearn.linear_model import LinearRegression


class Adult:
    filename = "./adult/adult.data"
    col_headers = np.array(["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                            "native-country"])
    col_dtypes = [float, str, float, str, float, str,
                  str, str, str, str, float, float, float,
                  str, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {1: ['Self-emp-inc', 'Federal-gov', 'Self-emp-not-inc', 'Private', 'Local-gov', 'State-gov', '?', 'Without-pay', 'Never-worked'],
                      3: ["Doctorate", "Prof-school", "Masters", "Bachelors", "Assoc-acdm", "Assoc-voc", "Some-college",
                          "HS-grad", "12th", "11th", "10th", "9th", "7th-8th", "5th-6th", "1st-4th", "Preschool"],
                      5: ['Married-civ-spouse', 'Married-AF-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'],
                      6: ['Exec-managerial', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Craft-repair', 'Adm-clerical', '?', 'Machine-op-inspct',
                          'Farming-fishing', 'Handlers-cleaners', 'Other-service', 'Priv-house-serv', 'Armed-Forces'],
                      7: ['Husband', 'Wife', 'Not-in-family', 'Unmarried', 'Own-child', 'Other-relative'],
                      }
    correl_coef = {0: 1, 1: -1, 2: 0, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: None, 9: None, 10: 1, 11: 1, 12: 1, 13: 0}

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array([1 if coef is not None else 0 for coef in correl_coef.values()])
    correl_coef = {c: coef if coef is not None else 0 for c, coef in correl_coef.items()}

    # Give each dataset its own csv reader
    def read_csv(self):
        """Read data from a CSV file into a 2D array."""
        data = []
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                cleaned_row = [entry.replace(' ', '') for entry in row]
                data.append(cleaned_row)
        return data


class Bank:
    # age: 18-95
    # job: ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    # marital: ['divorced', 'married', 'single']
    # education: ['primary', 'secondary', 'tertiary', 'unknown']
    # default: ['no', 'yes']
    # housing: ['no', 'yes']
    # loan: ['no', 'yes']
    # contact: ['cellular', 'telephone', 'unknown']
    # poutcome: ['failure', 'other', 'success', 'unknown']

    filename = './bank/train.csv'
    col_headers = np.array(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
                           )
    col_dtypes = [float, str, str, str, str, float,
                  str, str, str, float, str, float, float,
                  float, float, str, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {1: ['retired', 'student', 'management', 'technician', 'admin.', 'unknown', 'self-employed', 'services', 'entrepreneur', 'blue-collar', 'housemaid', 'unemployed'],
                      2: ['single', 'divorced', 'married'],
                      3: ['tertiary', 'secondary', 'unknown', 'primary'],
                      4: ['yes', 'no'],
                      6: ['yes', 'no'],
                      7: ['yes', 'no'],
                      8: ['cellular', 'unknown', 'telephone'],
                      10: ['mar', 'sep', 'oct', 'dec', 'jun', 'jul', 'aug', 'nov', 'may', 'feb', 'apr', 'jan'],
                      15: ['success',  'other', 'unknown', 'failure'],
                      }
    correl_coef = {0: 0, 1: -1, 2: -1, 3: -1, 4: 1, 5: 1, 6: 1, 7: 1, 8: -1, 9: 0, 10: -1, 11: 1, 12: 1, 13: 0, 14: 1, 15: -1}

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array([1 if coef is not None else 0 for coef in correl_coef.values()])
    correl_coef = {c: coef if coef is not None else 0 for c, coef in correl_coef.items()}

    def read_csv(self):
        """Read data from a CSV file into a 2D array."""
        data = []
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = row[0].split(';')
                cleaned_row = [entry.replace('"', '') for entry in row]
                data.append(cleaned_row)
        data = data[1:]
        return data


def map_strings_to_int(xs):
    """Convert a list of strings to a list of integers based on the unique and sorted order of the strings."""

    # Identify unique values and sort them
    unique_sorted_values = sorted(set(xs))

    # Create a dictionary that maps each unique string value to a unique integer
    value_to_int = {value: i for i, value in enumerate(unique_sorted_values)}

    # Convert the original list into a list of integers using the dictionary
    mapped_data = [value_to_int[item] for item in xs]

    return np.array(mapped_data, dtype=float)


class Dataset:
    def __init__(self, ds_prop):
        self.ds_prop = ds_prop
        data_2d_array = ds_prop.read_csv()[:-1]
        data = np.array(data_2d_array)

        self.data = data
        self.col_to_head = ds_prop.col_to_head
        self.col_to_dtype = ds_prop.col_to_dtype

        # Map all data to float
        numerical_data = []
        for j in range(data.shape[1]):
            col = data[:, j]
            if self.col_to_dtype[j] == str:
                float_col = map_strings_to_int(col)
            else:
                float_col = col.astype(float)
            numerical_data.append(float_col)

        numerical_data = np.stack(numerical_data).T
        self.num_data = numerical_data

        # Ordered categorical labels
        self.ordered_data = numerical_data.copy()
        for c in ds_prop.ordered_labels.keys():
            self.ordered_data[:, c] = self.str_to_order_int(c)

    # Map strings to ordered ints
    def str_to_order_int(self, col_no):
        print(f'Mapping column number {col_no}, {self.col_to_head[col_no]}')

        order_map = {s: i for i, s in enumerate(self.ds_prop.ordered_labels[col_no])}
        ordered_data = [order_map[s] for s in self.data[:, col_no]]
        ordered_data = np.array(ordered_data, dtype=float)

        return ordered_data

    def get_ordered(self, col_no):
        xs = self.ordered_data[:, col_no]
        if len(xs.shape) == 1:
            xs = xs[:, np.newaxis]

        mean, std = np.mean(xs, axis=0), np.std(xs, axis=0)
        xs = (xs - mean) / (std + 1e-3)
        return xs

    def get_base(self, col_no):
        xs = self.num_data[:, col_no]
        if len(xs.shape) == 1:
            xs = xs[:, np.newaxis]

        mean, std = np.mean(xs, axis=0), np.std(xs, axis=0)
        xs = (xs - mean) / (std + 1e-3)
        return xs

    def get_bias(self, col_no):
        return [self.ds_prop.correl_coef[c] for c in col_no], self.ds_prop.correl_mask[col_no]


def analyse_dataset(ds):
    data = ds.ordered_data

    col = 5

    print()
    print(ds.ds_prop.col_headers[col])
    print()
    xs = np.array(data[:, col], dtype=float).reshape(-1, 1)

    ys = map_strings_to_int(data[:, -1])
    ys = np.array(ys, dtype=float)

    model = LinearRegression()
    model.fit(xs, ys)

    print(f"Slope (Coefficient): {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print()
    print()

    for x in ds.ds_prop.ordered_labels[col]:
        idx = (ds.data[:, col] == x)
        print(f"{x}: {np.mean(ds.num_data[idx, -1])}")


if __name__ == "__main__":
    analyse_dataset(Dataset(Adult()))
