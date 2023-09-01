import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Logistic Regression Model
class LogisticRegression:
    def __init__(self, fit_intercept=True, bias=0, lam=0):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print(model.beta)
    print(beta)


if __name__ == "__main__":
    test_logistic_regression()
