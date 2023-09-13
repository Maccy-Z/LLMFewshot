import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from baselines import Model
from monaonic_net import MonatoneNet


class LogRegBias(nn.Module, Model):
    loss_fn: nn.BCELoss
    linear: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, fit_intercept, lr=0.01, steps=100, bias: float | torch.Tensor = 0., lam=0., mask=0.):
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

        # Weighting
        num_neg = (ys_meta == 0).sum().item()
        num_pos = (ys_meta == 1).sum().item()
        tot = num_pos + num_neg
        weight_neg = tot / (2 * num_neg + 1e-4)
        weight_pos = tot / (2 * num_pos + 1e-4)
        weights = torch.where(ys_meta == 1, weight_pos, weight_neg)

        self.loss_fn = nn.BCELoss(weight=weights)
        self.linear = nn.Linear(dims[1], 1, bias=self.fit_intercept)
        self.linear.weight.data.fill_(0.0)
        if self.fit_intercept:
            self.linear.bias.data.fill_(0.0)
        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr)

        for i in range(self.steps):
            self.optimizer.step(self._closure)

        # if self.fit_intercept:
        #     print(f'weights: {self.linear.weight.detach().numpy()[0]}, bias: {self.linear.bias.detach().numpy()[0]}')
        # else:
        #     print(f'Learned weights: {self.linear.weight.detach().numpy()[0]}')

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


class MonatoneLogReg(nn.Module, Model):
    loss_fn: nn.BCELoss = nn.BCELoss()
    linear: nn.Module
    optimizer: torch.optim.Optimizer

    monatone_map: nn.ModuleList
    n_cols: int
    unique_elements: list[torch.Tensor] = []
    unique_idx: list[torch.Tensor] = []
    zero_idx: list[int | None] = []
    need_zero: list[bool] = []

    def __init__(self, lr=0.01, steps=100):
        super().__init__()
        self.steps = steps
        self.lr = lr

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = torch.tensor(xs_meta, dtype=torch.float32), torch.tensor(ys_meta, dtype=torch.float32)
        ys_meta = ys_meta.unsqueeze(1) if ys_meta.dim() == 1 else ys_meta

        self.xs_meta, self.ys_meta = xs_meta, ys_meta
        dims = xs_meta.shape
        self.n_cols = dims[1]

        # Need to initalise networks after dims is known
        monatone_map = [MonatoneNet(1, 64, 1) for _ in range(self.n_cols)]
        self.monatone_map = nn.ModuleList(monatone_map)

        self.linear = nn.Linear(dims[1], 1)
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(0.0)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        # For each column, we need to find the unique values and use these for the monatone map
        for col in range(self.n_cols):
            unique, idx = torch.unique(self.xs_meta[:, col], sorted=True, return_inverse=True)

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

        for i in range(self.steps):
            loss = self.forward_step()
            if i % 10 == 0:
                print(f'{i}, Loss: {loss:.3g}')

    def forward_step(self):
        self.optimizer.zero_grad()

        # Pass the unique elements through the monatone map and reconstruct
        xs_meta = torch.empty_like(self.xs_meta)
        for col in range(self.n_cols):
            mapped_col, C, reg_loss = self.monatone_map[col].forward(self.unique_elements[col])

            # Integration constant, get the zero element and remove from tensor if a zero element was added
            zero_idx = self.zero_idx[col]
            zero_val = mapped_col[zero_idx]

            if self.need_zero[col]:
                mapped_col = torch.cat([mapped_col[:zero_idx], mapped_col[zero_idx + 1:]])

            # Subtract offset and add integration constant
            mapped_col = mapped_col - zero_val + C

            mapped_col = mapped_col[self.unique_idx[col]]
            xs_meta[:, col] = mapped_col

        # rand_idx = torch.randint(0, xs_meta.shape[0], size=(500,))
        xs_meta = xs_meta  # [rand_idx]
        ys_meta = self.ys_meta  # [rand_idx]
        # print(C)
        preds = self.forward(xs_meta)

        loss = self.loss_fn(preds, ys_meta)
        loss.backward()
        self.optimizer.step()

        return loss

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


def probit(x):
    return 1 / (1 + np.exp(-x))


def sample_instance(p):
    return np.random.choice([0, 1], size=1, p=[1 - p, p])


# Plot predictions of model
def plot_net(model: nn.Module, beta, model_beta):
    xs = torch.linspace(-5., 5, 50)

    with torch.no_grad():
        preds, C, _ = model.forward(xs)
        preds = (preds - preds[25] + C) * model_beta

    true_logit = beta * torch.relu(xs)

    plt.plot(xs, preds)
    plt.plot(xs, true_logit)
    plt.ylim([-7, 7])
    plt.show()


def test_logistic_regression():
    n_sample, n_feat = 1000, 5

    np.random.seed(0)

    # Generate random observations
    # X = np.random.normal(0., 4, size=[n_sample, n_feat])
    X = np.random.randint(-40, 40, size=[n_sample, n_feat])
    X = X / 20

    beta = np.random.normal(0, 1, size=n_feat)
    X_map = np.maximum(0, X)
    y_num = X_map @ beta
    y_num = probit(y_num)
    y = np.array([sample_instance(p) for p in y_num])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Create and train the model
    model = MonatoneLogReg(lr=0.02, steps=200)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred, _ = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print(f'{beta = }')
    print(f'{model.linear.weight}')
    print()

    # Plot output of a monatone net
    for i in range(5):
        monatone_model = model.monatone_map[i]
        plot_net(monatone_model, beta[i], model.linear.weight[0, i])


if __name__ == "__main__":
    test_logistic_regression()
