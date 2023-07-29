# Generate random splits of train and test.
import os
import random
import toml

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


base_dir = "./data"
datasets = [f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')]
random.shuffle(datasets)

splits = split_list(datasets, 4)

for i, split in enumerate(splits):
    split_nos = list(range(4))
    split_nos.remove(i)

    train_splits = []
    for n in split_nos:
        train_splits += splits[n]

    split = {"train": train_splits, "test": split}
    with open(f'./splits/{i}', "w") as f:
        toml.dump(split, f)


    print()
    print(split)