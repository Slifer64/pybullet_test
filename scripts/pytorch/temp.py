import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def trainInTorch():

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    a = torch.randn((), requires_grad=False)
    b = torch.randn((), requires_grad=False)
    c = torch.randn((), requires_grad=False)
    d = torch.randn((), requires_grad=False)

    lr = 1e-6

    for i in range(2000):

        y_pred = a + b*x + c*x**2 + d*x**3

        loss = (y_pred-y).pow(2).sum()

        if i % 200 == 199:
            print('iter %4d: loss = %.3f' % (i+1, loss.item()))

        y_pred_grad = 2*(y_pred - y)
        a_grad = torch.sum(y_pred_grad)
        b_grad = torch.sum(y_pred_grad * x)
        c_grad = torch.sum(y_pred_grad * x**2)
        d_grad = torch.sum(y_pred_grad * x**3)

        a -= lr*a_grad
        b -= lr*b_grad
        c -= lr*c_grad
        d -= lr*d_grad

    print('Finished!')
    print('y_pred = %.2f + %.2fx + %.2fx^2 + %.2fx^3' % (a.item(), b.item(), c.item(), d.item()))

    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), y_pred.detach().numpy(), color="blue", linestyle='-', linewidth=2, label='y_pred')
    ax.plot(x.detach().numpy(), y.detach().numpy(), color="magenta", linestyle='--', linewidth=2, label='y')
    ax.legend()
    plt.show()


def trainInTorchWithNN():

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    xx = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))

    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
    loss_criterion = torch.nn.MSELoss(reduction='sum')

    lr = 1e-6

    for i in range(2000):

        y_pred = model(xx)

        loss = loss_criterion(y_pred, y)

        if i % 200 == 199:
            print('iter %4d: loss = %.3f' % (i+1, loss.item()))

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr*param.grad

            # # zero all grads
            # for param in net.parameters():
            #     param.grad = None


    print('Finished!')
    a, b, c, d = model[0].bias, model[0].weight[0, 0], model[0].weight[0, 1], model[0].weight[0, 2]
    print('y_pred = %.2f + %.2fx + %.2fx^2 + %.2fx^3' % (a, b, c, d))

    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), y_pred.detach().numpy(), color="blue", linestyle='-', linewidth=2, label='y_pred')
    ax.plot(x.detach().numpy(), y.detach().numpy(), color="magenta", linestyle='--', linewidth=2, label='y')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    # trainInTorch()

    trainInTorchWithNN()