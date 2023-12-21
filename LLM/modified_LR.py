import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import time

from optim_baseline import Model
from monaonic_net import MonatoneNet


def probit(x):
    return 1 / (1 + np.exp(-x))


def sample_instance(p):
    return np.random.choice([0, 1], size=1, p=[1 - p, p])


class LogRegBias(nn.Module, Model):
    loss_fn: nn.BCELoss
    linear: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, fit_intercept, lr, steps, lam, bias: float | torch.Tensor, mask):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.steps = steps
        self.lr = lr
        self.bias, self.lam, self.mask = bias, lam, mask

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = torch.tensor(xs_meta, dtype=torch.float32), torch.tensor(ys_meta, dtype=torch.float32)
        ys_meta = ys_meta.unsqueeze(1) if ys_meta.dim() == 1 else ys_meta

        self.xs_meta, self.ys_meta = xs_meta, ys_meta
        dims = xs_meta.shape

        self.loss_fn = nn.BCELoss()
        self.linear = nn.Linear(dims[1], 1, bias=self.fit_intercept)
        self.linear.weight.data.fill_(0.0)
        if self.fit_intercept:
            self.linear.bias.data.fill_(0.0)
        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr)

        for i in range(self.steps):
            self.optimizer.step(self._closure)

    def _closure(self):
        self.optimizer.zero_grad()
        preds = self.forward(self.xs_meta)

        loss = self.loss_fn(preds, self.ys_meta) + self.lam * self._reg_loss()
        loss.backward()
        return loss

    def _reg_loss(self):
        reg_loss = torch.linalg.norm((self.linear.weight - self.bias) * self.mask) ** 2
        return reg_loss

    def predict(self, xs_target) -> np.array:
        xs_target = torch.tensor(xs_target, dtype=torch.float32)
        with torch.no_grad():
            preds = self.forward(xs_target)

        return preds > 0.5, preds

    def get_acc(self, xs_target, ys_target):
        y_pred, y_probs = self.predict(xs_target)

        # Metrics
        accuracy = accuracy_score(ys_target, y_pred)
        auc = roc_auc_score(ys_target, y_probs)

        return accuracy, auc

    def get_accuracy(self, batch):
        raise NotImplementedError


# Holds data, finds unique elements for faster ODE forward.
class DataHolder:
    def __init__(self, xs):
        self.xs = xs
        self.unique_elements = []
        self.unique_idx = []
        self.zero_idx = []
        self.need_zero = []

        self.n_cols = xs.shape[1]
        self.shape = xs.shape

        # For each column, we need to find the unique values and use these for the monatone map
        for col in range(self.n_cols):
            unique, idx = torch.unique(self.xs[:, col], sorted=True, return_inverse=True)

            # Need to include 0 in the unique elements as starting point of indefinite integral
            # Handle case where 0 is already in unique elements and not in
            if 0. in unique:
                self.need_zero.append(False)
            else:
                unique = torch.cat([torch.tensor([0.]), unique]).sort()[0]
                self.need_zero.append(True)

            zero_idx = torch.where(unique == 0)[0]
            self.zero_idx.append(zero_idx.item())

            self.unique_elements.append(unique), self.unique_idx.append(idx)

    # Return unique elemnets for ODE solver, including 0
    def get_unique(self, col):
        return self.unique_elements[col]

    # Reconstruct full sequence from mapped elements and indices
    def reconstruct(self, col, xs_mapped):
        mapped_s, unique_s = xs_mapped.shape[0], self.unique_elements[col].shape[0]
        assert mapped_s == unique_s | (mapped_s == unique_s + 1), \
            f'Mapped items must have same length as unique elements, {mapped_s=}, {unique_s=}'

        # Integration constant, get the zero element and remove from tensor if a zero element was added
        zero_idx = self.zero_idx[col]
        zero_val = xs_mapped[zero_idx]

        if self.need_zero[col]:
            xs_mapped = torch.cat([xs_mapped[:zero_idx], xs_mapped[zero_idx + 1:]])

        xs_mapped = xs_mapped - zero_val

        xs_mapped = xs_mapped[self.unique_idx[col]]
        return xs_mapped, zero_val


class MonatoneLogReg(nn.Module, Model):
    loss_fn: nn.BCELoss
    linear: nn.Module
    optimizer: torch.optim.Optimizer

    monatone_map: nn.ModuleList
    xs_holder: DataHolder
    n_cols: int

    def __init__(self, lr, steps, lam, bias: torch.Tensor):
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.lam = lam
        self.bias = bias

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = torch.tensor(xs_meta, dtype=torch.float32), torch.tensor(ys_meta, dtype=torch.float32)
        ys_meta = ys_meta.unsqueeze(1) if ys_meta.dim() == 1 else ys_meta

        self.xs_meta, self.ys_meta = xs_meta, ys_meta
        self.xs_holder = DataHolder(xs_meta)

        dims = xs_meta.shape
        self.n_cols = dims[1]

        # Need to initalise networks after dims is known
        monatone_map = [MonatoneNet(1, 64, 1) for _ in range(self.n_cols)]
        self.monatone_map = nn.ModuleList(monatone_map)

        self.linear = nn.Linear(dims[1], 1)
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(0.0)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        self.loss_fn = nn.BCELoss()

        for i in range(self.steps):
            self.train_step()

    # Return predictions from a batch of data
    def forward_step(self, xs_holder) -> np.array:
        beta_effs = []

        xs_monatone = torch.empty(xs_holder.shape)
        # Pass the unique elements through the monatone map and reconstruct
        for col in range(self.n_cols):
            xs_unique = xs_holder.get_unique(col)

            mapped_col = self.monatone_map[col].forward(xs_unique)

            xs_reconstruct, zero_val = xs_holder.reconstruct(col, mapped_col)
            xs_monatone[:, col] = xs_reconstruct

            # Get effective beta
            ys = mapped_col - zero_val
            ys = ys * self.linear.weight[0, col]
            # Stop div by zero by clamping
            xs_clamp = torch.clamp(xs_unique, min=0.0015) + torch.clamp(xs_unique, max=-0.001)#torch.clamp(xs_unique, min=0.01, max=-0.01).sign() * torch.clamp(xs_unique.abs(), min=0.001)
            beta_eff = torch.mean(ys / xs_clamp)

            beta_effs.append(beta_eff)


        # Multiply by weights
        preds = self.forward(xs_monatone)

        return preds, torch.stack(beta_effs)

    # Single optimisation step
    def train_step(self):
        self.optimizer.zero_grad()

        preds, beta_effs = self.forward_step(self.xs_holder)
        reg_loss = torch.sum((beta_effs - self.bias) ** 2)
        loss = self.loss_fn(preds, self.ys_meta) + reg_loss * self.lam
        loss.backward()
        self.optimizer.step()
        self.betas = beta_effs
        return loss

    # Return test accuracy and auc
    def get_acc(self, xs_target, ys_target):
        xs_target = torch.from_numpy(xs_target.astype(float))
        ys_target = torch.from_numpy(ys_target.astype(int).squeeze())

        xs_holder = DataHolder(xs_target)
        with torch.no_grad():
            preds, _ = self.forward_step(xs_holder)

        y_pred, y_probs = preds > 0.5, preds

        # Metrics
        accuracy = accuracy_score(ys_target, y_pred)
        auc = roc_auc_score(ys_target, y_probs)

        return accuracy, auc

    def get_accuracy(self, batch):
        raise NotImplementedError

    # Plot predictions of model
    def plot_net(self, get_col):
        cols = range(self.n_cols)
        for col in cols:
            xs = self.xs_holder.get_unique(col)
            mask = (xs != 0.)
            model = self.monatone_map[col]
            with torch.no_grad():
                preds = model.forward(xs)

                beta = self.linear.weight[0, col]
                preds = beta * preds

            if col == get_col:
                return xs[mask], preds[mask]


def test_logistic_regression():
    n_sample, n_feat = 1000, 5

    np.random.seed(1)

    # Generate random observations
    X = np.random.normal(0., 3, size=[n_sample, n_feat])
    # X = np.random.randint(-40, 40, size=[n_sample, n_feat])
    # X = X / 20

    beta = np.random.normal(0, 1, size=n_feat)
    X_map = np.maximum(0, X)
    y_num = X_map @ beta
    y_num = probit(y_num)
    y = np.array([sample_instance(p) for p in y_num])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Create and train the model
    model = MonatoneLogReg(lr=0.02, steps=50, lam=0.5, bias=0.)
    model.fit(X_train, y_train)

    # Make predictions
    accuracy, _ = model.get_acc(X_test, y_test)

    # Compute accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # print(f'{beta = }')
    # print(f'{model.linear.weight}')
    print()

    # Plot output of a monatone net
    for i in range(5):
        monatone_model = model.monatone_map[i]
        plot_net(monatone_model, beta[i], model.linear.weight[0, i], i)

    # model.plot_net()


if __name__ == "__main__":
    test_logistic_regression()
