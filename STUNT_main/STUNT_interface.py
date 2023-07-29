import time
import sys

import copy
import faiss
from train.metric_based import get_accuracy
import torch.nn.functional as F
from data.income import Income, Income2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# from Fewshot.comparison2 import Model
device = torch.device("cpu")


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


class MLPProto(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, out_features, bias=True)
        )

    def forward(self, inputs):
        # print(inputs.shape)
        embeddings = self.encoder(inputs)
        return embeddings


def protonet_step(step, model, optimizer, batch):

    train_inputs, train_targets = batch['train']
    num_ways = len(set(list(train_targets[0].numpy())))
    test_inputs, test_targets = batch['test']

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)

    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets, num_ways)
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
    loss = F.cross_entropy(-squared_distances, test_targets)

    """ outer gradient step """
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()
    if step % 20 == 0:
        print()
        print((loss.item()))
        print(acc)


def main(device):
    lr = 0.001
    model_size = (105, 1024, 1024)
    steps = 100
    num_shots = 5
    seed = 0
    bs = 4
    num_ways = 5
    n_shot_test = 5

    """ define dataset and dataloader """
    train_loader = Income(
                            shot=num_shots,
                            tasks_per_batch=bs,
                            test_num_way=num_ways,
                            query=n_shot_test)

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    model = MLPProto(*model_size).to(device)
    model.train()

    params = model.parameters()
    optimizer = optim.Adam(params, lr=lr)
    print("Starting training ")
    for step in range(1, steps + 1):
        train_batch = next(train_loader)

        protonet_step(step, model, optimizer, train_batch)

    torch.save(model, "/STUNT_main/logs/model.pt")

class STUNT_utils:
    model: torch.nn.Module
    optim: torch.optim.Adam
    steps: int
    shot: int
    tasks_per_batch: int
    test_num_way: int
    query: int
    kmeans_iter: int

    def protonet_step(self, batch):

        train_inputs, train_targets = batch['train']
        num_ways = len(set(list(train_targets[0].numpy())))
        test_inputs, test_targets = batch['test']

        train_embeddings = self.model(train_inputs)
        test_embeddings = self.model(test_inputs)

        prototypes = self.get_prototypes(train_embeddings, train_targets, num_ways)
        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                       - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
        loss = F.cross_entropy(-squared_distances, test_targets)

        """ outer gradient step """
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        _, preds = torch.min(squared_distances, dim=-1)


    @staticmethod
    def get_prototypes(embeddings, targets, num_classes):
        """Compute the prototypes (the mean vector of the embedded training/support
        points belonging to its class) for each classes in the task.
        Parameters
        ----------
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the support points. This tensor
            has shape `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the support points. This tensor has
            shape `(batch_size, num_examples)`.
        num_classes : int
            Number of classes in the task.
        Returns
        -------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        """
        batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

#         print(targets, num_classes)
        num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
        indices = targets.unsqueeze(-1).expand_as(embeddings)
        prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

        return prototypes

    def get_batch(self, x):
        # x.shape = [ds_rows, num_targs]
        xs, ys, xq, yq = [], [], [], []
        assert 2 * (self.shot + self.query) <= x.shape[0]
        num_way = self.test_num_way
        n_shot, n_query = self.shot, self.query
        for _ in range(self.tasks_per_batch):

            # Need at least shot + query examples of each class.
            # This might not always be possible, so allow for smaller amounts.
            loop_count = 0
            while True:
                # Number of columns to select from
                min_col = max(int(x.shape[1] * 0.2), 1)
                max_col = int(x.shape[1] * 0.5)

                if min_col == max_col:
                    max_col += 1

                col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]
                # Select random columns and mask
                task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)
                # kmeans selected columns and get pseudo-labels
                kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=self.kmeans_iter, nredo=1,
                                      verbose=False, min_points_per_centroid=self.shot + self.query)
                kmeans.train(masked_x)

                D, I = kmeans.index.search(masked_x, 1) # Distance / nearest cluster
                y = I[:, 0].astype(np.int32)
                class_list, counts = np.unique(y, return_counts=True)
                min_count = min(counts)

                if min_count >= (self.shot + self.query):

                    break
                elif loop_count > 5:
                    n_query = min(n_query, 1)
                    n_shot = min(min_count -1, n_shot)
                    break
                else:
                    loop_count += 1

            # Dataset cannot be split into classes.
            if n_shot == 0 or len(counts) == 1:
                raise NameError("Cannot split dataset")

            num_to_permute = x.shape[0]
            for t_idx in task_idx:
                rand_perm = np.random.permutation(num_to_permute)
                x[:, t_idx] = x[:, t_idx][rand_perm]
            classes = np.random.choice(class_list, num_way, replace=False)

            # Select min_shot postivie and negative samples
            support_idx, query_idx = [], []
            for k in classes:
                k_idx = np.where(y == k)[0]
                permutation = np.random.permutation(len(k_idx))
                k_idx = k_idx[permutation]
                support_idx.append(k_idx[:n_shot])
                query_idx.append(k_idx[n_shot:n_shot + n_query])

            support_idx = self.interleave(*support_idx)
            query_idx = self.interleave(*query_idx)

            support_x = x[support_idx]
            query_x = x[query_idx]
            s_y = y[support_idx]
            q_y = y[query_idx]

            #support_set.append(support_x)
            #support_sety.append(support_y)
            # query_set.append(query_x)
            # query_sety.append(query_y)

            #print(np.array(support_x).shape)
            #xs_k = np.concatenate(support_set, 0)
            #print(xs_k.shape)
            #xq_k = np.concatenate(query_set, 0)
            #ys_k = np.concatenate(support_sety, 0)
            # yq_k = np.concatenate(query_sety, 0)

            xs_k = np.array(support_x)
            xq_k = np.array(query_x)
            ys_k = np.array(s_y)
            yq_k = np.array(q_y)


            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs = [x[:n_shot * 2] for x in xs]
        ys = [y[:n_shot * 2] for y in ys]
        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [self.tasks_per_batch, num_way * n_shot, -1])

        xq = np.reshape(
            xq,
            [self.tasks_per_batch, num_way * n_query, -1])

        xs = xs.astype(np.float32)
        xq = xq.astype(np.float32)
        ys = ys.astype(np.float32)
        yq = yq.astype(np.float32)

        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)

        ys = torch.from_numpy(ys).type(torch.LongTensor)
        yq = torch.from_numpy(yq).type(torch.LongTensor)

        batch = {'train': (xs, ys), 'test': (xq, yq)}
        return batch

    @staticmethod
    def interleave(arr1, arr2):
        c = np.empty((arr1.size + arr2.size,), dtype=arr2.dtype)
        c[0::2] = arr1
        c[1::2] = arr2

        return c

    def __repr__(self):
        return "STUNT"



def eval():
    data_name = "income"
    shot_num = 1
    load_path = ''
    seed = 0
    input_size = 105
    output_size = 2
    hidden_dim = 1024


    model = torch.load("/home/maccyz/Documents/STUNT-main/logs/model.pt").cpu().eval()
    # model = MLPProto(input_size, hidden_dim, hidden_dim)
    # model.load_state_dict(load_model.state_dict())

    train_x = np.load('./data/income/xtrain.npy')
    train_y = np.load('./data/income/ytrain.npy')
    test_x = np.load('./data/income/xtest.npy')
    test_y = np.load('./data/income/ytest.npy')
    train_idx = np.load('./data/income/index{}/train_idx_{}.npy'.format(shot_num, seed))

    few_train = model(torch.tensor(train_x[train_idx]).float())
    support_x = few_train.detach().numpy()
    support_y = train_y[train_idx]

    few_test = model(torch.tensor(test_x).float())
    query_x = few_test.detach().numpy()
    query_y = test_y

    def get_accuracy(prototypes, embeddings, targets):
        sq_distances = torch.sum((prototypes.unsqueeze(1)
                                  - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        return torch.mean(predictions.eq(targets).float()) * 100.

    train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
    train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
    val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
    prototypes = get_prototypes(train_x, train_y, output_size)
    acc = get_accuracy(prototypes, val_x, val_y).item()

    print(seed, acc)


if __name__ == "__main__":
    """ argument define """
    # eval()
    # exit(2)
    main("cpu")
    #stunt = STUNT()

#    stunt.fit()
