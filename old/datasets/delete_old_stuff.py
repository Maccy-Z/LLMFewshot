# Delete old baselines and batches

import os

def delete_if_exists(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        print(f'{f} doesnt exist')
        return

    print(f, "deleted")
    return

data_dir = "data"

datasets = sorted([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')])

for ds in datasets:
    delete_if_exists(f'{data_dir}/{ds}/base_RF_fix.dat')
    delete_if_exists(f'{data_dir}/{ds}/base_fix_num_1s.dat')
    delete_if_exists(f'{data_dir}/{ds}/baselines.dat')

    try:
        for f in os.listdir(f'{data_dir}/{ds}/batches'):
            delete_if_exists(f'{data_dir}/{ds}/batches/{f}')
    except FileNotFoundError as e:
        print(e)



