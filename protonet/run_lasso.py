import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from sklearn.linear_model import Lasso, LogisticRegression

from config import Config
from dataloader import SplitDataloader


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
                x = F.tanh(layer(x) * 1)
                # print(torch.mean(x), torch.std(x))

            # No activation after the last layer
            x = self.layers[-1](x)
        return x


class ProtoNet:
    batch_clf: list
    batch_masks: list

    def __init__(self, cfg):
        self.cfg = cfg

        self.embed_model = SimpleMLP(cfg, [14, 10000, 20000])
        self.embed_model.to('cuda')

    # Generate latent embeddings
    def _to_embedding(self, xs):
        xs = torch.stack(xs)  # shape = [BS, N, N_cols]
        xs = xs.to('cuda')
        xs = self.embed_model.forward(xs)  # shape = [BS, N, embed_dim]
        return xs

    # Given meta embeddings and labels, generate prototypes
    def fit(self, xs_meta, ys_metas):
        embed_metas = self._to_embedding(xs_meta)

        # Seperate out batches
        self.batch_clf, self.batch_masks = [], []
        for embed_meta, ys_meta in zip(embed_metas, ys_metas, strict=True):
            # clf = LogisticRegression(penalty='l1', C=1, solver='liblinear') #
            lasso_clf = Lasso(alpha=0.01)
            lasso_clf.fit(embed_meta.cpu(), ys_meta.cpu())

            coefs = lasso_clf.coef_.squeeze()

            selected_mask = coefs != 0
            embed_meta = embed_meta[:, selected_mask]
            clf = LogisticRegression()
            clf.fit(embed_meta.cpu(), ys_meta)

            self.batch_clf.append(clf)
            self.batch_masks.append(selected_mask)

    # Compare targets to prototypes
    def predict_proba(self, xs_targ):
        targ_embeds = self._to_embedding(xs_targ)  # shape = [BS, N_targ, embed_dim]

        # Loop over batches
        all_probs = []
        for clf, mask, targ_embed in zip(self.batch_clf, self.batch_masks, targ_embeds, strict=True):
            # Get predictions using Lasso
            targ_embed = targ_embed[:, mask]

            probs = clf.predict_proba(targ_embed.cpu())

            # Convert to probabilities
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs)
        all_probs = torch.from_numpy(all_probs)
        return all_probs


def main(cfg: Config, nametag=None):
    dl = SplitDataloader(cfg, dataset='adult', all_cols=True)

    print()
    print("Training data names:", dl)

    model = ProtoNet(cfg=cfg)

    # Baseline Model
    # cb_base = BasicModel("KNN")

    accs = []
    base_mean, base_std = 0., 0.
    for batch in itertools.islice(dl, cfg.N_batches):
        xs_meta, ys_meta, xs_target, ys_target, max_N_label = batch

        model.fit(xs_meta, ys_meta)
        ys_pred_targ = model.predict_proba(xs_target)

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
