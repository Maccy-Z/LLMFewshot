from re import I
import numpy as np
import torch
import os
import copy
import faiss


class Income(object):
    def __init__(self, shot, tasks_per_batch, test_num_way, query):
        super().__init__()
        self.num_classes = 2
        self.shot = shot
        self.query = query
        self.tasks_per_batch = tasks_per_batch
        self.unlabeled_x = np.load('./data/income/train_x.npy')
        self.test_num_way = test_num_way


    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        xs, ys, xq, yq = [], [], [], []
        x = self.unlabeled_x
        num_way = self.test_num_way

        for _ in range(self.tasks_per_batch):
            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            tmp_x = copy.deepcopy(x)
            min_count = 0
            while min_count < (self.shot + self.query):
                # Number of columns to select from
                min_col = int(x.shape[1] * 0.2)
                max_col = int(x.shape[1] * 0.5)
                col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]

                # Select random columns and mask
                task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)

                # kmeans selected columns and get pseudo-labels
                kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False, min_points_per_centroid=self.shot + self.query, gpu=1)
                kmeans.train(masked_x)
                D, I = kmeans.index.search(masked_x, 1) # Distance / nearest cluster
                y = I[:, 0].astype(np.int32)
                class_list, counts = np.unique(y, return_counts=True)
                min_count = min(counts)

            num_to_permute = x.shape[0]
            for t_idx in task_idx:
                rand_perm = np.random.permutation(num_to_permute)
                tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

            classes = np.random.choice(class_list, num_way, replace=False)

            support_idx = []
            query_idx = []
            for k in classes:
                k_idx = np.where(y == k)[0]
                permutation = np.random.permutation(len(k_idx))
                k_idx = k_idx[permutation]
                support_idx.append(k_idx[:self.shot])
                query_idx.append(k_idx[self.shot:self.shot + self.query])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)

            support_x = tmp_x[support_idx]
            query_x = tmp_x[query_idx]
            s_y = y[support_idx]
            q_y = y[query_idx]
            support_y = copy.deepcopy(s_y)
            query_y = copy.deepcopy(q_y)

            i = 0
            for k in classes:
                support_y[s_y == k] = i
                query_y[q_y == k] = i
                i += 1
            support_set.append(support_x)
            support_sety.append(support_y)
            query_set.append(query_x)
            query_sety.append(query_y)

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)


        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [self.tasks_per_batch, num_way * self.shot, -1])

        xq = np.reshape(
            xq,
            [self.tasks_per_batch, num_way * self.query, -1])

        xs = xs.astype(np.float32)
        xq = xq.astype(np.float32)
        ys = ys.astype(np.float32)
        yq = yq.astype(np.float32)

        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)

        ys = torch.from_numpy(ys).type(torch.LongTensor)
        yq = torch.from_numpy(yq).type(torch.LongTensor)
        print(xs.shape)
        batch = {'train': (xs, ys), 'test': (xq, yq)}
        return batch

class Income2(object):
    def __init__(self, shot, tasks_per_batch, test_num_way, query):
        super().__init__()
        self.num_classes = 2
        self.shot = shot
        self.query = query
        self.tasks_per_batch = tasks_per_batch
        # self.unlabeled_x = np.load('./STUNT-main/data/income/train_x.npy')
        self.test_num_way = test_num_way


    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self, x):
        xs, ys, xq, yq = [], [], [], []
        # x = self.unlabeled_x
        num_way = self.test_num_way

        for _ in range(self.tasks_per_batch):

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            tmp_x = copy.deepcopy(x)
            min_count = 0
            while min_count < (self.shot + self.query):
                # Number of columns to select from
                min_col = int(x.shape[1] * 0.2)
                max_col = int(x.shape[1] * 0.5)
                col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]

                # Select random columns and mask
                task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)

                # kmeans selected columns and get pseudo-labels
                kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False, min_points_per_centroid=self.shot + self.query, gpu=1)
                kmeans.train(masked_x)
                D, I = kmeans.index.search(masked_x, 1) # Distance / nearest cluster
                y = I[:, 0].astype(np.int32)
                class_list, counts = np.unique(y, return_counts=True)
                min_count = min(counts)

            num_to_permute = x.shape[0]
            for t_idx in task_idx:
                rand_perm = np.random.permutation(num_to_permute)
                tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

            classes = np.random.choice(class_list, num_way, replace=False)

            support_idx = []
            query_idx = []
            for k in classes:
                k_idx = np.where(y == k)[0]
                permutation = np.random.permutation(len(k_idx))
                k_idx = k_idx[permutation]
                support_idx.append(k_idx[:self.shot])
                query_idx.append(k_idx[self.shot:self.shot + self.query])
            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)

            support_x = tmp_x[support_idx]
            query_x = tmp_x[query_idx]
            s_y = y[support_idx]
            q_y = y[query_idx]
            support_y = copy.deepcopy(s_y)
            query_y = copy.deepcopy(q_y)

            i = 0
            for k in classes:
                support_y[s_y == k] = i
                query_y[q_y == k] = i
                i += 1

            support_set.append(support_x)
            support_sety.append(support_y)
            query_set.append(query_x)
            query_sety.append(query_y)

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [self.tasks_per_batch, num_way * self.shot, -1])

        xq = np.reshape(
            xq,
            [self.tasks_per_batch, num_way * self.query, -1])

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

