import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def plotResults(x, y, y_pred):
    fig, ax = plt.subplots()
    ax.plot(x, y_pred, label='y_pred', color='magenta', linestyle='-', linewidth=2)
    ax.plot(x, y, label='y', color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel('input x')
    ax.set_ylabel('output y')
    ax.set_title('sine fitting')
    ax.legend()
    plt.show()


class MyMSELoss:

    def __init__(self, reduction='sum'):

        if reduction == 'sum':
            self.reduction_fun = lambda y_pred, y: (y - y_pred).pow(2).sum()
        elif reduction == 'mean':
            self.reduction_fun = lambda y_pred, y: (y - y_pred).pow(2).mean()
        else:
            raise RuntimeError('Invalid reduction: %s' % reduction)

    def __call__(self, y_pred, y):

        return self.reduction_fun(y_pred, y)


class MyOptimizer:

    def __init__(self, parameters, learning_rate):
        self.parameters = [param for param in parameters]
        self.lr = learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                param -= self.lr * param.grad


class Polynomial:  # (torch.nn.Module):
    # if we inherit from nn.Module, then:
    # - we don't need to implement __call__() and parameters()
    # - we have to define the self.coeff as nn.Parameter, so that it gets registered in the nn.Modules parameters
    # forward() is redundant if we don't inherit from nn.Module

    def __init__(self, n):
        super().__init__()
        self.n = n
        # self.coeff = torch.nn.Parameter(torch.randn((1, self.n + 1)))
        self.coeff = torch.randn((1, self.n + 1), requires_grad=True)

    def forward(self, x):
        y = self.coeff[:, 0] * torch.ones_like(x)
        xi = torch.ones_like(x)
        for i in range(1, self.n+1):
            xi = xi * x
            y = y + self.coeff[:, i]*xi
        return y

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        yield self.coeff  # we only have one nn.Parameter

    def string(self):
        coeff = [self.coeff[:, i].item() for i in range(self.n+1)]
        y_str = '{0:.1f}'.format(coeff[0])
        if self.n > 0:
            y_str += ' + {0:.1f}x'.format(coeff[1])
        for i in range(2, self.n+1):
            y_str += ' + {0:.1f}x^{1:d}'.format(coeff[i], i)

        return y_str


class PolynomialNN(torch.nn.Module):

    def __init__(self, n):
        super().__init__()

        self.n = n
        # nn.Sequential contains other Modules, and applies them in sequence
        # The Linear Module holds internal Tensors for its weight and bias.
        # The Flatten layer flattens the output of the linear layer to a 1D tensor
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 1, bias=True),  # bias:a, n params ai, each for ai*x^i, i=1...n
            torch.nn.Flatten(0, 1)  # (first dim to flatten, last dim to flatten)
        )

    def forward(self, x):
        # the output y is a linear function of (x, x^2, x^3), so we can consider it as a linear layer neural network.
        xx = x.unsqueeze(-1).pow(torch.arange(start=1, end=self.n+1, step=1))
        return self.net(xx)

    def string(self):
        linear_layer = self.net[0]
        coeff = [linear_layer.bias.item()] + [linear_layer.weight[:, i].item() for i in range(self.n)]
        y_str = '{0:.1f}'.format(coeff[0])
        if self.n > 0:
            y_str += ' + {0:.1f}x'.format(coeff[1])
        for i in range(2, self.n+1):
            y_str += ' + {0:.1f}x^{1:d}'.format(coeff[i], i)

        return y_str


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.
        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.a + self.b * x + self.c * x**2 + self.d * x**3
        for exp in range(4, np.random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        return 'y = {0:.1f} + {1:.1f}x + {2:.1f}x^2 + {3:.1f}x^3 + ({3:.1f}x^4)? + ({3:.1f}x^5)?'.\
            format(self.a.item(), self.b.item(), self.c.item(), self.d.item(), self.e.item(), self.e.item())

def trainInNumpy():
    x = np.linspace(-np.pi, np.pi, 2000)
    y = np.sin(x)

    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    lr = 1e-6

    for i in range(3000):

        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = np.square(y_pred - y).sum()

        grad_y_pred = 2 * (y_pred - y)
        grad_a = np.sum(grad_y_pred)
        grad_b = np.sum(grad_y_pred * x)
        grad_c = np.sum(grad_y_pred * x ** 2)
        grad_d = np.sum(grad_y_pred * x ** 3)

        a -= lr * grad_a
        b -= lr * grad_b
        c -= lr * grad_c
        d -= lr * grad_d

        if i % 100 == 0:
            print('%5d: loss = %.2f' % (i, loss))

    print('Finished!')

    plotResults(x, y, y_pred)


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

    plotResults(x.numpy(), y.numpy(), y_pred.detach().numpy())


def trainInTorchWithNN():
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # =======  Model  =======
    model = Polynomial(3)
    # model = PolynomialNN(3)
    # model = DynamicNet()

    # =======  Loss function  =======
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # loss_fn = lambda y_pred, y: (y_pred - y).pow(2).sum()
    loss_fn = MyMSELoss(reduction='sum')

    # =======  Optimizer  =======
    # optimizer = MyOptimizer(model.parameters(), learning_rate=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    prev_loss = math.inf
    for t in range(14000):

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute loss.
        loss = loss_fn(y_pred, y)

        if math.fabs(prev_loss - loss.item()) < 1e-4:
            break
        prev_loss = loss.item()

        if t % 100 == 99:
            print('%4d: loss = %6.2f' % (t + 1, loss.item()))

        # Zero the gradients before running the backward pass.
        # model.zero_grad()
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights
        optimizer.step()

    # For linear layer, its parameters are stored as `weight` and `bias`.
    print('Result:', model.string())

    plotResults(x.numpy(), y.numpy(), y_pred.detach().numpy())


if __name__ == '__main__':
    # trainInNumpy()

    # trainInTorch()

    trainInTorchWithNN()

    # trainInTorchWithNNandOptim()
