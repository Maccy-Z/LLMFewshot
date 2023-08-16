import torch
import torch.nn as nn
import torch_geometric as pyg
import numpy as np
import itertools
import time

from GAtt_Func import GATConvFunc
from save_holder import SaveHolder
from config import Config
from dataloader import SplitDataloader, d2v_pairer


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


# Dataset2vec model
class SetSetModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        h_size = cfg.d2v_h_dim
        f_depth = cfg.f_depth

        pos_enc_bias = cfg.pos_enc_bias
        pos_enc_dim = cfg.pos_enc_dim
        pos_depth = cfg.pos_enc_depth

        self.relu = nn.ReLU()

        # f network
        self.fs = ResBlock(2, h_size, h_size, n_blocks=f_depth)

        # Positional embedding Network
        self.ps = nn.ModuleList([])
        for _ in range(pos_depth - 1):
            self.ps.append(nn.Linear(h_size, h_size))

        self.p_out = nn.Linear(h_size, pos_enc_dim, bias=(pos_enc_bias != "off"))

        if pos_enc_bias == "zero":
            print(f'Positional encoding bias init to 0')
            self.p_out.bias.data.fill_(0)

    def forward_layers(self, x, splits):
        # x.shape = [BS*[N_rows], N_cols, 2]
        # f network
        x, _ = self.fs(x)  # [BS*[N_rows], N_cols, h_size]

        # Mean over each batch
        N_col = x.shape[1]
        unbatch_x = torch.split(x, splits)
        x = [torch.mean(x, dim=0) for x in unbatch_x]
        x = torch.cat(x)

        # Positional Encoding
        for layer in self.ps:
            x = self.relu(layer(x))
        pos_enc_out = self.p_out(x)

        pos_enc_out = pos_enc_out.view(-1, N_col, pos_enc_out.shape[-1])

        return pos_enc_out

    def forward(self, xs):
        splits = [len(x) for x in xs]
        xs = torch.cat(xs)
        pos_encs = self.forward_layers(xs, splits=splits)

        return pos_encs


class GNN2(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.GATConv = GATConvFunc()

        gat_heads = cfg.gat_heads
        gat_hid_dim = cfg.gat_hid_dim
        gat_in_dim = cfg.gat_in_dim
        gat_out_dim = cfg.gat_out_dim
        gat_layers = cfg.gat_layers

        self.gat_layers = nn.ModuleList([])
        self.gat_layers.append(pyg.nn.GATConv(gat_in_dim, gat_hid_dim, heads=gat_heads))
        for _ in range(gat_layers - 2):
            self.gat_layers.append(pyg.nn.GATConv(gat_hid_dim * gat_heads, gat_hid_dim, heads=gat_heads))
        self.gat_layers.append(pyg.nn.GATConv(gat_hid_dim * gat_heads, gat_out_dim, heads=gat_heads))

        self.lin_1 = torch.nn.Linear(gat_out_dim, cfg.proto_dim)

    # Generate additional fixed embeddings / graph
    def graph_matrix(self, N_rows, N_col):
        # A single densely connected graph
        nodes = torch.arange(N_col)
        interleave = torch.repeat_interleave(nodes, N_col)
        repeat = torch.tile(nodes, (N_col,))
        base_edge_idx = torch.stack([interleave, repeat])

        tot_repeats = sum(N_rows)
        edge_idx = torch.tile(base_edge_idx, [tot_repeats, 1, 1])
        offsets = torch.arange(tot_repeats).view(-1, 1, 1) * N_col

        edge_idx = edge_idx + offsets

        edge_idx = edge_idx.permute(1, 0, 2)
        edge_idx = edge_idx.reshape(2, -1)

        return edge_idx

    def forward(self, batch_xs, batch_pos_enc):
        """
        :param batch_xs:              shape = [BS][N_row, N_col]
        :param batch_pos_enc:         shape = [BS][N_row, enc_dim]
        :return output:               shape = [BS][N_row, N_col]
        """
        N_rows, batch_inputs = [], []
        for xs, pos_enc in zip(batch_xs, batch_pos_enc):
            N_row, N_col = xs.shape
            N_rows.append(N_row)

            # Concatenate pos_enc onto every row of xs
            pos_enc = pos_enc.repeat(N_row, 1, 1)
            xs = xs.unsqueeze(-1)
            xs = torch.cat([xs, pos_enc], dim=-1)
            xs = xs.view(-1, xs.size(2))

            batch_inputs.append(xs)

        # Inputs are stacked over all batches.
        batch_inputs = torch.cat(batch_inputs, dim=0)

        # Edges are fully connected graph for each row. Rows are processed independently.
        edge_idx = self.graph_matrix(N_rows, N_col=N_col)

        xs = batch_inputs
        # Forward each GAT layer
        for layer in self.gat_layers:
            xs = layer(xs, edge_idx)

        # Sum GAT node outputs for final predictions.
        xs = xs.view(-1, N_col, xs.size(-1))
        xs = xs.split(N_rows)
        batch_outputs = []
        for x in xs:
            x = torch.mean(x, dim=-2)
            x = self.lin_1(x)

            batch_outputs.append(x)

        return batch_outputs


class ProtoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gnn_model = GNN2(cfg=cfg)

    # From observations, generates latent embeddings
    def to_embedding(self, xs, pos_enc):
        # Pass through GNN -> average -> final linear
        embeddings = self.gnn_model(xs, pos_enc)
        return embeddings

    # Given meta embeddings and labels, generate prototypes
    def gen_prototypes(self, xs_meta, ys_metas, pos_enc):
        embed_metas = self.to_embedding(xs_meta, pos_enc)

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
    def forward(self, xs_targ, pos_enc, max_N_label):
        targ_embeds = self.to_embedding(xs_targ, pos_enc)

        # Loop over batches
        all_probs = []
        for protos, targ_embed in zip(self.batch_protos, targ_embeds, strict=True):
            # Find distance between each prototype by tiling one and interleaving the other.
            # tile prototype: [1, 2, 3 ,1, 2, 3]
            # repeat targets: [1, 1, 2, 2, 3, 3]

            labels, prototypes = protos.keys(), protos.values()
            labels, prototypes = list(labels), list(prototypes)

            N_class = len(protos)
            N_targs = len(targ_embed)

            prototypes = torch.stack(prototypes)
            prototypes = torch.tile(prototypes, (N_targs, 1))

            test = torch.repeat_interleave(targ_embed, N_class, dim=-2)

            # Calc distance and get probs
            distances = -torch.norm(test - prototypes, dim=-1)
            distances = distances.reshape(N_targs, N_class)
            probs = torch.nn.Softmax(dim=-1)(distances)

            # Probs are in order of protos.keys(). Map to true classes.
            true_probs = torch.zeros([self.cfg.N_target, max_N_label], dtype=torch.float32)
            true_probs[:, labels] = probs

            all_probs.append(true_probs)

        all_probs = torch.concatenate(all_probs)
        return all_probs


class ModelHolder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.d2v_model = SetSetModel(cfg=cfg)
        self.protonet = ProtoNet(cfg=cfg)

    # Forward Meta set and train
    def forward_meta(self, xs_meta, ys_meta):
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        pos_enc = self.d2v_model(pairs_meta)

        self.protonet.gen_prototypes(xs_meta, ys_meta, pos_enc)
        return pos_enc

    def forward_target(self, xs_target, pos_enc, max_N_label):
        preds = self.protonet.forward(xs_target, pos_enc, max_N_label)

        return preds.view(-1, max_N_label)

    def loss_fn(self, preds, targs):
        targs = torch.cat(targs)
        cross_entropy = torch.nn.functional.cross_entropy(preds, targs)
        return cross_entropy


def main(cfg, nametag=None):
    save_holder = None

    dl = SplitDataloader(cfg,
                         bs=cfg.bs, datasets=cfg.ds_group, testing=False)
    val_dl = SplitDataloader(cfg,
                             bs=1, datasets=cfg.ds_group, testing=True)

    print()
    print("Training data names:", dl)
    print("\nTest data names:", val_dl)

    model = ModelHolder(cfg=cfg)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.w_decay)

    accs, losses = [], []
    val_accs, val_losses = [], []
    st = time.time()
    for epoch in range(cfg.epochs):
        duration = time.time() - st
        st = time.time()
        print()
        print(f'{epoch = }, {duration = :.2g}s')

        save_grads = None

        # Train loop
        model.train()
        for xs_meta, ys_meta, xs_target, ys_target, max_N_label in itertools.islice(dl, cfg.ep_len):
            # First pass with the meta-set, train d2v and get embedding.
            pos_enc = model.forward_meta(xs_meta, ys_meta)

            # Second pass using previous embedding and train weight encoder
            ys_pred_targ = model.forward_target(xs_target, pos_enc, max_N_label)

            loss = model.loss_fn(ys_pred_targ, ys_target)
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Accuracy recording
            ys_target = torch.cat(ys_target)
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = torch.eq(predicted_labels, ys_target).sum().item() / len(ys_target)

            accs.append(accuracy), losses.append(loss.item())

            # Save train gradients
            grads = {n: torch.abs(p.grad) for n, p in model.named_parameters() if p.requires_grad and not p.grad is None}
            if save_grads is None:
                save_grads = grads
            else:
                for name, abs_grad in grads.items():
                    save_grads[name] += abs_grad

        print(f"Training accuracy : {np.mean(accs[-cfg.ep_len:]) * 100:.2f}%")

        # Validation loop
        model.eval()
        epoch_accs, epoch_losses = [], []
        for xs_meta, ys_meta, xs_target, ys_target, max_N_label in itertools.islice(dl, cfg.ep_len):
            with torch.no_grad():
                pos_enc = model.forward_meta(xs_meta, ys_meta)
                ys_pred_targ = model.forward_target(xs_target, pos_enc, max_N_label)
                loss = model.loss_fn(ys_pred_targ, ys_target)

            # Accuracy recording
            ys_target = torch.cat(ys_target)
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = torch.eq(predicted_labels, ys_target).sum().item() / len(ys_target)

            epoch_accs.append(accuracy), epoch_losses.append(loss.item())

        val_losses.append(epoch_losses), val_accs.append(epoch_accs)
        print(f'Validation accuracy: {np.mean(val_accs[-1]) * 100:.2f}%')

        # Average gradients
        for name, abs_grad in save_grads.items():
            save_grads[name] = torch.div(abs_grad, cfg.ep_len)

        # Save stats
        if save_holder is None:
            save_holder = SaveHolder(".", nametag=nametag, cfg=cfg)
        history = {"accs": accs, "loss": losses, "val_accs": val_accs, "val_loss": val_losses, "epoch_no": epoch}
        save_holder.save_model(model, optim, epoch=epoch)
        save_holder.save_history(history)
        save_holder.save_grads(save_grads)


if __name__ == "__main__":
    # torch.manual_seed(0)
    tag = ""  # input("Description: ")

    for test_no in range(1):
        print("---------------------------------")
        print("Starting test number", test_no)

        main(cfg=Config(), nametag=tag)

    print("")
    print(tag)
    print("Training Completed")
