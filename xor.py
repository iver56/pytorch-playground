# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.autograd import Variable


if __name__ == '__main__':
    batch_size = 1
    input_dim = 2
    hidden_dim = 10
    output_dim = 1

    # Dataset
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    x = torch.from_numpy(x)
    x = x.float()
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    y = torch.from_numpy(y)
    y = y.float()

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(x)
    y = Variable(y)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
        torch.nn.Sigmoid()
    )
    loss_fn = torch.nn.BCELoss()  # binary cross-entropy loss

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algorithms. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 0.03
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.data[0])

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    # Use the model
    output = model.forward(x)
    print(output.data.numpy())
