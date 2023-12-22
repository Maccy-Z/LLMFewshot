import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import time

from config import Config
from dataloader import SplitDataloader
from baselines import BasicModel


class ResBlock(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_blocks, out_relu=True):
        super().__init__()
        self.out_relu = out_relu

        self.res_modules = nn.ModuleList([])
        self.lin_in = nn.Linear(in_size, hid_size)

        for _ in range(n_blocks - 2):
            self.res_modules.append(nn.Linear(hid_size, hid_size))

        self.lin_out = nn.Linear(hid_size, out_size)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_in(x)
        for layer in self.res_modules:
            x_r = self.act(layer(x))
            x = x + x_r

        if self.out_relu:
            out = self.act(self.lin_out(x))
        else:
            out = self.lin_out(x)
        return out, x


class SimpleMLP(nn.Module):
    def __init__(self, cfg, layer_sizes):
        super(SimpleMLP, self).__init__()

        # Fix RNG init
        torch.manual_seed(cfg.seed)

        self.layers = nn.ModuleList()

        # Create layers based on the sizes provided
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        for l in self.layers:
            nn.init.kaiming_normal_(l.weight, nonlinearity='tanh')

    def forward(self, x):
        with torch.no_grad():
            # Pass data through each layer except for the last one
            for layer in self.layers[:-1]:
                x = F.tanh(layer(x) * 0.3)
                print(torch.mean(x), torch.std(x))

            # No activation after the last layer
            x = self.layers[-1](x)
        return x


class ProtoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed_model = SimpleMLP(cfg, [14, 10000, 10000, 20000])
        self.embed_model.to('cuda')

    # From observations, generates latent embeddings
    def to_embedding(self, xs):
        xs = torch.stack(xs)        # shape = [BS, N, N_cols]
        xs = xs.to('cuda')
        xs = self.embed_model.forward(xs)   # shape = [BS, N, embed_dim]
        return xs #.cpu()

    # Given meta embeddings and labels, generate prototypes
    def gen_prototypes(self, xs_meta, ys_metas):
        embed_metas = self.to_embedding(xs_meta)

        # Seperate out batches
        self.batch_protos = []
        for embed_meta, ys_meta in zip(embed_metas, ys_metas, strict=True):
            labels = torch.unique(ys_meta)
            embed_protos = [embed_meta[ys_meta == i] for i in labels]

            prototypes = {}
            for embed_proto, label in zip(embed_protos, labels, strict=True):
                prototype = torch.mean(embed_proto, dim=0)
                prototypes[label.item()] = prototype

            self.batch_protos.append(prototypes)

    # Compare targets to prototypes
    def forward(self, xs_targ, max_N_label):
        targ_embeds = self.to_embedding(xs_targ)  # shape = [BS, N_targ, embed_dim]

        # Loop over batches
        all_probs = []
        for protos, targ_embed in zip(self.batch_protos, targ_embeds, strict=True):
            # Find distance between each prototype by tiling one and interleaving the other.
            # tile prototype: [1, 2, 3 ,1, 2, 3]
            # repeat targets: [1, 1, 2, 2, 3, 3]

            labels, prototypes = protos.keys(), protos.values()
            labels, prototypes = list(labels), list(prototypes)  # prototypes.shape = [N_class, embed_dim]

            N_class = len(protos)
            N_targs = len(targ_embed)

            prototypes = torch.stack(prototypes)
            prototypes = torch.tile(prototypes, (N_targs, 1))  # Shape = [N_targ * N_class, embed_dim]

            test = torch.repeat_interleave(targ_embed, N_class, dim=-2)  # Shape = [N_targ * N_class, embed_dim]

            # Calc distance and get probs
            distances = -torch.norm(test - prototypes, dim=-1)
            # distances = torch.dot(test, prototypes.T)
            distances = distances.reshape(N_targs, N_class)
            probs = torch.nn.Softmax(dim=-1)(distances)

            # Probs are in order of protos.keys(). Map to true classes.
            true_probs = torch.zeros([N_targs, max_N_label], dtype=torch.float32, device='cuda')

            true_probs[:, labels] = probs

            all_probs.append(true_probs)

        all_probs = torch.concatenate(all_probs)

        return all_probs


class ModelHolder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.protonet = ProtoNet(cfg=cfg)

    # Forward Meta set and train
    def forward_meta(self, xs_meta, ys_meta):
        self.protonet.gen_prototypes(xs_meta, ys_meta)

    def forward_target(self, xs_target, max_N_label):
        preds = self.protonet.forward(xs_target, max_N_label)

        return preds.view(-1, max_N_label).cpu()


def main(cfg: Config, nametag=None):
    dl = SplitDataloader(cfg, dataset='adult', all_cols=True)

    print()
    print("Training data names:", dl)

    model = ModelHolder(cfg=cfg)

    # Baseline Model
    # cb_base = BasicModel("KNN")

    accs = []
    base_mean, base_std = 0., 0.
    for batch in itertools.islice(dl, cfg.N_batches):
        xs_meta, ys_meta, xs_target, ys_target, max_N_label = batch

        # Eval model
        model.forward_meta(xs_meta, ys_meta)
        ys_pred_targ = model.forward_target(xs_target, max_N_label)
        # Accuracy
        ys_target = torch.cat(ys_target)
        predicted_labels = torch.argmax(ys_pred_targ, dim=1)
        accuracy = torch.eq(predicted_labels, ys_target).numpy()
        accs += [accuracy]

        # Baseline accuracy
        # base_mean, base_std = cb_base.get_accuracy(batch)

    # print(f'Baseline accuracy: {base_mean * 100:.2f}% +- {base_std * 100:.2f}%')
    accs = np.concatenate(accs)
    acc, std = np.mean(accs), np.std(accs) / np.sqrt(accs.shape[0])

    print(f"Training accuracy : {acc * 100:.2f} +- {std * 100:.2f}")


if __name__ == "__main__":
    tag = ""  # input("Description: ")

    for test_no in range(1):
        print("---------------------------------")
        print("Starting test number", test_no)

        main(cfg=Config(), nametag=tag)

    print("")
    print(tag)
    print("Training Completed")
