import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn


class TorchLogReg(nn.Module):
    loss_fn: nn.BCELoss
    linear: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, fit_intercept, lr=0.01, steps=100, bias: float | torch.Tensor = 0., lam=0.):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.steps = steps
        self.lr = lr
        self.bias, self.lam = bias, lam

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

        # print(preds, ys_meta)

        loss = self.loss_fn(preds, self.ys_meta) + self.lam * self._reg_loss()
        loss.backward()
        return loss

    def _reg_loss(self):
        reg_loss = torch.linalg.norm(self.linear.weight - self.bias) ** 2

        return reg_loss

    def predict(self, xs_target) -> np.array:
        xs_target = torch.tensor(xs_target, dtype=torch.float32)
        with torch.no_grad():
            preds = self.forward(xs_target)

        return preds > 0.5


# Logistic Regression Model
class LogisticRegression:
    def __init__(self, fit_intercept=True, bias=0., lam=0.):
        self.fit_intercept = fit_intercept
        self.beta = None
        self.bias = bias
        self.lam = lam

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, beta, X, y, bias):
        m = len(y)
        p = self.sigmoid(X @ beta)
        epsilon = 1e-5
        cost = -(1 / m) * (y.T @ np.log(p + epsilon) + (1 - y).T @ np.log(1 - p + epsilon)) + self.lam * np.linalg.norm(bias - beta)
        return cost

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        # Initialize beta
        self.beta = np.zeros(X.shape[1])

        # Optimize using L-BFGS optimizer
        result = minimize(self.cost_function, self.beta, args=(X, y, 0), method='L-BFGS-B')
        self.beta = result.x

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(X @ self.beta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


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
