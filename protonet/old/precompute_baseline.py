import torch
import pickle
import os
import csv

from baselines import TabnetModel, FTTrModel, BasicModel, STUNT
from dataloader import SplitDataloader
from precompute_batches import load_batch
data_dir = './datasets/data'


def main_append(f, num_targets):

    models = [BasicModel("R_Forest")]

    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        print(model)
        for num_rows in [3,5,10,15]:
            for num_cols in [-3,]:
                try:
                    batch = load_batch(ds_name=f, N_meta=num_rows, N_target=num_targets)
                except IndexError as e:
                    print(e)
                    break
                mean_acc, std_acc = model.get_accuracy(batch)
                model_accs.append([model, num_rows, num_cols, mean_acc, std_acc])

    with open(f'{data_dir}/{f}/base_RF_fix.dat', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in model_accs:
            writer.writerow(row)


def main(f, num_targets, num_1s="a"):

    models = [
              BasicModel("LR") , BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              # TabnetModel(),
              FTTrModel(),
              ]

    model_accs = [] # Save format: [model, num_rows, num_cols, (num_1s), acc, std]

    for model in models:
        print(model)
        for num_rows in [3,5,10,15]:
            for num_cols in [-3,]:
                try:
                    batch = load_batch(ds_name=f, num_rows=num_rows, num_cols=-3, num_targets=num_targets, num_1s=num_1s)
                except IndexError as e:
                    print(e)
                    break
                mean_acc, std_acc = model.get_accuracy(batch)
                model_accs.append([model, num_rows, num_cols, num_1s, mean_acc, std_acc])

    with open(f'{data_dir}/{f}/base_RF_fix.dat', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "num_rows", "num_cols", "acc", "std"])
        for row in model_accs:
            writer.writerow(row)

if __name__ == "__main__":
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    num_bs = 200
    num_targs = 5

    files = [f for f in sorted(os.listdir(data_dir)) if os.path.isdir(f'{data_dir}/{f}')]
    for f in files:
        print("---------------------")
        print(f)

        #save_batch(f, num_bs, num_targs)
        main_append(f, num_targets=num_targs)
        # main(f, num_targets=num_targs)
