import torch

from main import ModelHolder
import numpy as np
from config import load_config

max_batches = 200

class FLAT():
    def __init__(self, load_no, save_ep=None):
        save_dir = f'./saves/save_{load_no}'
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')

        cfg = load_config(f'{save_dir}/config.toml')

        self.model = ModelHolder(cfg=cfg)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def get_accuracy(self, batch):
        xs_metas, ys_metas, xs_targets, ys_targets, _ = batch
        accs = []

        xs_metas, ys_metas, xs_targets, ys_targets = xs_metas[:max_batches], ys_metas[:max_batches], xs_targets[:max_batches], ys_targets[:max_batches]

        self.fit(xs_metas, ys_metas)
        a = self.get_acc(xs_targets, ys_targets)

        accs.append(a)

        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])
        return mean, std

    def fit(self, xs_meta, ys_meta):
        self.unique_ys_meta = np.unique(ys_meta)

        with torch.no_grad():
            self.pos_enc = self.model.forward_meta(xs_meta, ys_meta)

    def get_acc(self, xs_target, ys_target) -> np.array:
        unique_target = np.unique(ys_target)
        unique_labels = np.union1d(self.unique_ys_meta, unique_target)
        max_N_label = np.max(unique_labels) + 1

        with torch.no_grad():
            ys_pred_targ = self.model.forward_target(xs_target, self.pos_enc, max_N_label)
        predicted_labels = torch.argmax(ys_pred_targ, dim=1)

        return torch.eq(predicted_labels, ys_target.flatten()).numpy()

    def __repr__(self):
        return "FLAT"


class FLAT_batch():
    def __init__(self, load_no, save_ep=None):
        save_dir = f'./saves/save_{load_no}'
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')

        cfg = load_config(f'{save_dir}/config.toml')

        self.model = ModelHolder(cfg=cfg)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def get_accuracy(self, batch):
        xs_metas, ys_metas, xs_targets, ys_targets, _ = batch
        accs = []

        batch_no = 0
        for xs_meta, xs_target, ys_meta, ys_target in zip(xs_metas, xs_targets, ys_metas, ys_targets):
            self.fit(xs_meta, ys_meta)
            a = self.get_acc(xs_target, ys_target)

            accs.append(a)

            if batch_no > max_batches:
                break

        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])
        return mean, std

    def fit(self, xs_meta, ys_meta):
        self.unique_ys_meta = np.unique(ys_meta)

        with torch.no_grad():
            self.pos_enc = self.model.forward_meta([xs_meta], [ys_meta])

    def get_acc(self, xs_target, ys_target) -> np.array:
        unique_target = np.unique(ys_target)
        unique_labels = np.union1d(self.unique_ys_meta, unique_target)
        max_N_label = np.max(unique_labels) + 1

        with torch.no_grad():
            ys_pred_targ = self.model.forward_target([xs_target], self.pos_enc, max_N_label)
        predicted_labels = torch.argmax(ys_pred_targ, dim=1)

        return torch.eq(predicted_labels, ys_target.flatten()).numpy()

    def __repr__(self):
        return "FLAT"



if __name__ == "__main__":
    torch.manual_seed(0)

    bs, n_rows, n_cols = 200, 10, 5
    xs_metas = [torch.randn([n_rows, n_cols]) for _ in range(bs)]
    ys_metas = [torch.randint(0, 2, [n_rows]) for _ in range(bs)]
    xs_targets = [torch.randn([n_rows, n_cols]) for _ in range(bs)]
    ys_targets = torch.randint(0, 2, [bs, n_rows])

    model = FLAT(57)
    acc = model.get_accuracy((xs_metas, ys_metas, xs_targets, ys_targets, None))
    print(acc)

    model = FLAT_batch(57)
    acc = model.get_accuracy((xs_metas, ys_metas, xs_targets, ys_targets, None))
    print(acc)
    # (0.5, 0.04096159602595202)
