import torch
from torchvision import transforms


from data.income import Income

def get_meta_dataset(dataset, num_shots=1, n_shot_test=15, num_ways=10, bs=4, seed=0, only_test=False):
    if dataset == 'income':
        meta_train_dataset = Income(tabular_size = 105,
                                    seed=seed,
                                    source='train',
                                    shot=num_shots,
                                    tasks_per_batch=bs,
                                    test_num_way = num_ways,
                                    query = n_shot_test)

        # meta_val_dataset = Income(tabular_size = 105,
        #                             seed=P.seed,
        #                             source='val',
        #                             shot=1,
        #                             tasks_per_batch=P.test_batch_size,
        #                             test_num_way = 2,
        #                             query = 30)

    else:
        raise NotImplementedError()

    return meta_train_dataset, None# , meta_val_dataset
