import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn

from baselines import Model


class TorchLogReg(nn.Module, Model):
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
        #weights = torch.ones_like(weights)

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


def probit(x):
    return 1 / (1 + np.exp(-x))


def sample_instance(p):
    return np.random.choice([0, 1], size=1, p=[1 - p, p])


def test_logistic_regression():
    n_sample, n_feat = 10000, 5

    # Generate random observations
    X = np.random.normal(0., 1, size=[n_sample, n_feat])
    beta = np.random.normal(0, 1, size=n_feat)
    y_num = X @ beta
    y_num = probit(y_num)
    y = np.array([sample_instance(p) for p in y_num])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Create and train the model
    # model = LogisticRegression(fit_intercept=False)
    model = TorchLogReg(fit_intercept=False, lr=0.5, steps=1000, bias=torch.tensor([0, 0, 1., 0.5, -1]), lam=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'{beta = }')


if __name__ == "__main__":
    test_logistic_regression()
