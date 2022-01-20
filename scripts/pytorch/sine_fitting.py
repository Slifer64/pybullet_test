import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def trainInNumpy():

    x = np.linspace(-np.pi, np.pi, 2000)
    y = np.sin(x)

    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    lr = 1e-6

    for i in range(3000):

        y_pred = a + b*x + c*x**2 + d*x**3
        loss = np.square(y_pred-y).sum()

        grad_y_pred = 2*(y_pred - y)
        grad_a = np.sum(grad_y_pred)
        grad_b = np.sum(grad_y_pred * x)
        grad_c = np.sum(grad_y_pred * x**2)
        grad_d = np.sum(grad_y_pred * x**3)

        a -= lr*grad_a
        b -= lr * grad_b
        c -= lr * grad_c
        d -= lr * grad_d

        if i % 100 == 0:
            print('%5d: loss = %.2f' % (i, loss))

    print('Finished!')

    fig, ax = plt.subplots()
    ax.plot(x, y_pred, label='y_pred', color='magenta', linestyle='-', linewidth=2)
    ax.plot(x, y, label='y', color='blue', linestyle='--', linewidth=2)
    ax.legend()
    plt.show()


def trainInTorch():

    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)

    a = torch.randn((), requires_grad=True)
    b = torch.randn((), requires_grad=True)
    c = torch.randn((), requires_grad=True)
    d = torch.randn((), requires_grad=True)

    lr = 1e-6

    for i in range(3000):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        loss = (y_pred - y).pow(2).sum()
        loss.backward()

        if i % 100 == 0:
            print('%5d: loss = %.2f' % (i, loss.item()))

        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
            c -= lr * c.grad
            d -= lr * d.grad

            a.grad, b.grad, c.grad, d.grad = None, None, None, None

    print('Finished!')

    fig, ax = plt.subplots()
    ax.plot(x.numpy(), y_pred.detach().numpy(), label='y_pred', color='magenta', linestyle='-', linewidth=2)
    ax.plot(x.numpy(), y.numpy(), label='y', color='blue', linestyle='--', linewidth=2)
    ax.legend()
    plt.show()

def trainInTorchWithNN():
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # For this example, the output y is a linear function of (x, x^2, x^3), so
    # we can consider it as a linear layer neural network. Let's prepare the
    # tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
    # (3,), for this case, broadcasting semantics will apply to obtain a tensor
    # of shape (2000, 3)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. The Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # The Flatten layer flatens the output of the linear layer to a 1D tensor,
    # to match the shape of `y`.
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1) # (first dim to flatten, last dim to flatten)
    )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-6
    for t in range(2000):

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(xx)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print('%4d: loss = %6.2f' % (t+1, loss.item()))

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # You can access the first layer of `model` like accessing the first item of a list
    linear_layer = model[0]

    # For linear layer, its parameters are stored as `weight` and `bias`.
    print('Result: y = %.1f + %.1f x + %.1f x^2 + %.1f x^3' % (linear_layer.bias.item(),
           linear_layer.weight[:, 0].item(), linear_layer.weight[:, 1].item(), linear_layer.weight[:, 2].item()))


def trainInTorchWithNNandOptim():
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Prepare the input tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for t in range(2000):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(xx)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print('%4d: loss = %6.2f' % (t+1, loss.item()))

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

    # You can access the first layer of `model` like accessing the first item of a list
    linear_layer = model[0]
    # For linear layer, its parameters are stored as `weight` and `bias`.
    print('Result: y = %.1f + %.1f x + %.1f x^2 + %.1f x^3' % (linear_layer.bias.item(),
           linear_layer.weight[:, 0].item(), linear_layer.weight[:, 1].item(), linear_layer.weight[:, 2].item()))


if __name__ == '__main__':

    # trainInNumpy()

    # trainInTorch()

    # trainInTorchWithNN()

    trainInTorchWithNNandOptim()