import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

from matplotlib import pyplot as plt
import random
import numpy as np


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.fc2.bias.data.fill_(0.0)
        #self.fc2.weight.data.fill_(0.0)
        self.act = torch.nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# Enforce loss on ouput of model if output is negative
def neg_loss(output):
    return torch.relu(-output)


class MonatoneNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.mlp_model = MLP(input_size, hidden_size, output_size)

    # Return predicted outputs and regularisation loss
    def forward(self, xs):

        if self.training:
            output = odeint(self.diffeq, torch.tensor([0.], requires_grad=False), xs,
                            method='midpoint', options={"step_size": 0.1})
            # method="dopri8", rtol=0.05, atol=0.05, options={"max_num_steps": 25, "dtype": torch.float32})
        else:
            output = odeint(self.diffeq, torch.tensor([0.]), xs,
                            method="rk4", options={"step_size": 0.05})
        output = output.squeeze()

        return output

    def diffeq(self, x, y):
        diff = self.mlp_model.forward(x.unsqueeze(0))
        grad = diff + 1
        return torch.nn.Softplus()(grad) - 0.5


def train(model, xs, ys):
    # Initialize the model, loss, and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # Train the model
    epochs = 500

    losses, reg_losses = [], []
    for epoch in range(epochs):
        # Forward pass
        outputs, reg_loss, _ = model(xs)
        loss = criterion(outputs, ys)
        full_loss = loss + 0.25 * reg_loss

        # Backward pass and optimization
        full_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())
        reg_losses.append(reg_loss.detach())

        # Print the loss every 500 epochs
        if (epoch + 1) % 100 == 0:
            loss = np.mean(losses)
            reg_loss = np.mean(reg_losses)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}')
            losses = []

    return model


def main():
    import time

    monat_model = MonatoneNet(1, 32, 1)
    xs = torch.linspace(0., 1, 10)
    ys = 3 * (xs - 0.8 * xs ** 2)

    train(monat_model, xs, ys)

    xs_test = torch.linspace(0., 1, 5000).unsqueeze(-1)

    with torch.no_grad():
        st = time.time()
        y_pred, _, _ = monat_model(xs_test.squeeze())
        print(f'{time.time() - st:.3g}')

    plt.scatter(xs, ys)
    plt.plot(xs_test, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
