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
        self.fc2.weight.data.fill_(0.0)
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
        self.init_param = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.vals = []

    # Return predicted outputs and regularisation loss
    def forward(self, xs):
        self.vals = []

        output = odeint(self.diffeq, torch.tensor([0.]), xs,
                        #method="midpoint", options={"step_size": 0.05})
                        method="dopri8", rtol=0.1, atol=0.1, options={"max_num_steps":25, "dtype":torch.float32})
        output = output.squeeze()

        reg_loss = None# neg_loss(torch.cat(self.vals)).mean()
        return output, self.init_param, reg_loss

    def diffeq(self, x, y):
        diff = self.mlp_model.forward(x.unsqueeze(0))
        grad = diff #+ 1
        #self.vals.append(grad)
        return torch.nn.Softplus()(grad) #- 0.1


def train(model, xs, ys):
    # Initialize the model, loss, and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    # Train the model
    epochs = 500

    losses, reg_losses = [], []
    for epoch in range(epochs):
        # Forward pass
        outputs, reg_loss = model(xs)
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
    monat_model = MonatoneNet(1, 32, 1)
    xs = torch.linspace(0., 1, 10)
    ys = 3 * (xs - 0.8 * xs ** 2)

    train(monat_model, xs, ys)

    xs_test = torch.linspace(0.01, 1, 50).unsqueeze(-1)

    ys_pred = []
    for x in xs_test:
        with torch.no_grad():
            y_pred, _ = monat_model(x)
        ys_pred.append(y_pred)

    plt.scatter(xs, ys)
    plt.plot(xs_test, ys_pred)
    plt.show()


if __name__ == "__main__":
    main()
