from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from datasets import Adult, Dataset

ds = Dataset(Adult())

xs = ds.get_onehot([1, 2, 3])
ys = ds.num_data[:, -1]



print(xs[:, 9])


