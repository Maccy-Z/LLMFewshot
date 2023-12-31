import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.io import arff


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

        return data[:-1]


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
    col_headers = np.array(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                            'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'])
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
                      8: ['cellular', 'telephone', 'unknown'],
                      10: ['mar', 'sep', 'oct', 'dec', 'jun', 'jul', 'aug', 'nov', 'may', 'feb', 'apr', 'jan'],
                      15: ['success', 'other', 'unknown', 'failure'],
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


# DONE
class Blood:
    filename = "./blood/blood.csv"
    col_headers = np.array(
        [
            "Recency (months)",
            "Frequency (times)",
            "Monetary (c.c. blood)",
            "Time (months)",
        ]
    )
    col_dtypes = [int, int, int, int, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {

    }
    # GPT: "Recency in months since the last donation positively correlates with the probability of donating blood. More recent donors are more likely to donate again."
    # => towards 0 is more recent, so positive correlation with variable is actually negative
    correl_coef = {
        0: -1,
        1: 1,
        2: 1,
        3: 1,
    }

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = pd.read_csv(self.filename)  # .iloc[:, :-1]
        return data


# DONE
class California:
    filename = "./california/housing.csv"
    col_headers = np.array(
        [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
            "median_house_value",
        ]
    )
    col_dtypes = []

    # ChatGPT-4 generated orderings:
    ordered_labels = {
        8: ['ISLAND', 'NEAR OCEAN', 'NEAR BAY', '<1H OCEAN', 'INLAND']
    }
    correl_coef = {
        0: 0,
        1: 0,
        2: -1,
        3: 1,
        4: 1,
        5: -1,
        6: 1,
        7: 1,
        8: -1
    }

    # Process data

    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = pd.read_csv(self.filename)
        data["temp"] = data["median_house_value"]
        data = data.drop(columns='median_house_value').rename(columns={'temp': 'median_house_value'})

        self.col_dtypes = map_dtypes_to_py_types(data.dtypes.values)
        self.col_headers = data.columns
        self.col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        self.col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}

        median = data['median_house_value'].median()
        data['median_house_value'] = data['median_house_value'].apply(lambda x: 1 if x > median else 0)
        data = data.dropna()
        #[print(c) for c in data.columns]
        return data


# DONE
class Car:
    filename = "car/car_evaluation.csv"
    col_headers = np.array(
        [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
        ]
    )

    col_dtypes = [str, str, int, int, str, str]

    # ChatGPT-4 generated orderings:
    ordered_labels = {
        0: ["high", "med", "low", "vhigh"],
        1: ["high", "med", "low", "vhigh"],
        2: ["4", "5more", "3", "2"],
        3: ["more", "4", "2"],
        4: ["big", "med", "small"],
        5: ["high", "med", "low"],
    }
    correl_coef = {
        0: 1,
        1: -1,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
    }

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = pd.read_csv(self.filename)
        # data.drop("state", axis=1, inplace=True)  # drop the label
        self.col_dtypes = map_dtypes_to_py_types(data.dtypes.values)
        self.col_headers = data.columns
        self.col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        self.col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}
        #print(data.to_numpy())
        return data


# DONE
class Diabetes:
    filename = "./diabetes/diabetes.csv"
    col_headers = np.array(
        [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ]
    )

    col_dtypes = []

    # ChatGPT-4 generated orderings:
    ordered_labels = {

    }
    correl_coef = {
        0: 1,
        1: 1,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
    }

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = pd.read_csv(self.filename)
        #data.drop("Outcome", axis=1, inplace=True)  # drop the label
        self.col_dtypes = map_dtypes_to_py_types(data.dtypes.values)
        self.col_headers = data.columns
        self.col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        self.col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}
        return data


# DONE
class Heart:
    filename = "./heart/heart.csv"
    col_headers = np.array(
        [
            "Age",
            "Sex",
            "ChestPainType",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "RestingECG",
            "MaxHR",
            "ExerciseAngina",
            "Oldpeak",
            "ST_Slope",
        ]
    )

    col_dtypes = []

    # ChatGPT-4 generated orderings:
    ordered_labels = {
        2: ["TA", "ATA", "NAP", "ASY"],
        6: ["LVH", "ST", "Normal"],
        10: ["Down", "Flat", "Up"],
    }
    correl_coef = {
        0: 1,  #
        1: None,  #
        2: 0,  #
        3: 1,  #
        4: 1,  # #
        5: 1,  ##
        6: 0,  ##
        7: 0,  #
        8: 1,  #
        9: 1,  #
        10: -1,
    }

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = pd.read_csv(self.filename)
        #data.drop('HeartDisease', axis=1, inplace=True)  # drop the label
        self.col_dtypes = map_dtypes_to_py_types(data.dtypes.values)
        self.col_headers = data.columns
        self.col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        self.col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}

        return data


# DONE
class Jungle:
    filename = './jungle/jungle.arff'
    col_headers = np.array(
        [
            "white_piece0_strength",
            "white_piece0_file",
            "white_piece0_rank",
            "black_piece0_strength",
            "black_piece0_file",
            "black_piece0_rank",
        ]
    )

    col_dtypes = []

    # ChatGPT-4 generated orderings:
    ordered_labels = {

    }
    correl_coef = {
        0: 1,
        1: 0,
        2: 1,
        3: -1,
        4: -1,
        5: 1,
    }

    # Process data
    col_to_head = {i: h for i, h in enumerate(col_headers)}
    col_to_dtype = {i: d for i, d in enumerate(col_dtypes)}
    correl_mask = np.array(
        [1 if coef is not None else 0 for coef in correl_coef.values()]
    )
    correl_coef = {
        c: coef if coef is not None else 0 for c, coef in correl_coef.items()
    }

    def read_csv(self):
        data = arff.loadarff(self.filename)
        data = pd.DataFrame(data[0])
        #data.drop('class', axis=1, inplace=True)  # drop the label
        self.col_dtypes = map_dtypes_to_py_types(data.dtypes.values)
        self.col_headers = data.columns
        self.col_to_head = {i: h for i, h in enumerate(self.col_headers)}
        self.col_to_dtype = {i: d for i, d in enumerate(self.col_dtypes)}

        # Binarise to white win
        data["class"] = data["class"] == b"w"
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


def map_dtypes_to_py_types(column_dtypes):
    dtype_to_python_type = {
        np.dtype('int32'): int,
        np.dtype('float32'): float,
        np.dtype('int64'): int,
        np.dtype('float64'): float,
        np.dtype('O'): str,  # 'O' is for 'object', usually string
    }

    return [dtype_to_python_type[dtype] for dtype in column_dtypes]


class Dataset:
    def __init__(self, ds_prop):
        self.ds_prop = ds_prop
        data_2d_array = ds_prop.read_csv()#[:-1]
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


    def get_unnormaliesd(self, col_no):
        xs = self.data[:, col_no]
        if len(xs.shape) == 1:
            xs = xs[:, np.newaxis]

        return xs

    def get_ordered(self, col_no):
        xs = self.ordered_data[:, col_no]
        if len(xs.shape) == 1:
            xs = xs[:, np.newaxis]

        scalar = StandardScaler().fit(xs)
        xs = scalar.transform(xs)
        return xs

    def get_base(self, col_no):
        xs = self.num_data[:, col_no]
        if len(xs.shape) == 1:
            xs = xs[:, np.newaxis]

        scalar = StandardScaler().fit(xs)
        xs = scalar.transform(xs)
        return xs

    def get_bias(self, col_no):
        return [self.ds_prop.correl_coef[c] for c in col_no], self.ds_prop.correl_mask[col_no]

    # Return one-hot encoding of categorical columns
    def get_onehot(self, col_nos):
        xs_one_hot = []
        for col_no in col_nos:
            if col_no in self.ds_prop.ordered_labels.keys():
                xs = self.data[:, col_no]
                xs = xs[:, np.newaxis]
                encoder = OneHotEncoder(sparse_output=False)
                xs_one = encoder.fit_transform(xs)
                xs_one_hot.append(xs_one)
            else:
                xs_one_hot.append(self.get_base([col_no]))

        xs_one_hot = np.concatenate(xs_one_hot, axis=1)
        return xs_one_hot

    def __len__(self):
        return self.data.shape[1] - 1


def balanced_batches(X, y, bs, num_batches, seed=0):
    """
    Generator function that yields balanced batches from dataset X with labels y.
    Each batch will have total_batch_size samples, distributed as evenly as possible among classes.

    Parameters:
    - X: Features in the dataset.
    - y: Labels corresponding to X.
    - total_batch_size: Total number of samples required in the batch.
    - rng: A numpy random number generator instance for controlled sampling.

    Yields:
    - X_batch: Features of the sampled batch.
    - y_batch: Labels corresponding to X_batch.
    """

    RNG = np.random.default_rng(seed)

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    # Calculate samples per class and determine the "extra" samples
    samples_per_class = bs // n_classes
    extra_samples = bs % n_classes

    batches = []
    for _ in range(num_batches):
        batch_indices = []
        for idx, label in enumerate(unique_labels):
            label_indices = np.where(y == label)[0]

            # Adjust samples for this class if there are extra samples
            current_samples = samples_per_class + 1 if idx < extra_samples else samples_per_class

            if len(label_indices) < current_samples:
                raise ValueError(f"Label {label} has fewer samples than the requested samples for this class.")

            sampled_indices = RNG.choice(label_indices, current_samples, replace=False)
            batch_indices.extend(sampled_indices)

        remainder_indices = np.setdiff1d(np.arange(len(X)), batch_indices)

        X_batch, y_batch = X[batch_indices], y[batch_indices]
        X_remainder, y_remainder = X[remainder_indices], y[remainder_indices]

        batches.append((X_batch, X_remainder, y_batch, y_remainder))
    return batches


def analyse_dataset(ds, col):
    data = ds.ordered_data

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
        print(f"{x}: {np.mean(ds.num_data[idx, -1]):.3g}")


if __name__ == "__main__":
    ds = Dataset(California())
    x = ds.get_unnormaliesd(range(9))
    print(x[1])

    analyse_dataset(ds, col=7)
