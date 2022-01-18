import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update

    # from torch import nn, optim
    #
    # model = torchvision.models.resnet18(pretrained=True)
    #
    # # Freeze all the parameters in the network
    # for param in model.parameters():
    #     param.requires_grad = False

    # model = torchvision.models.resnet18(pretrained=True)
    # data = torch.rand(1, 3, 64, 64)
    # labels = torch.rand(1, 1000)
    #
    # prediction = model(data)  # forward pass
    #
    # loss = (prediction - labels).sum()
    # loss.backward()  # backward pass
    #
    # optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #
    # optim.step()  # gradient descent

    # a = torch.tensor([2., 3.], requires_grad=True)
    # b = torch.tensor([6., 4.], requires_grad=True)
    # Q = 3 * a ** 3 - b ** 2
    # # loss = Q.sum()
    # # loss.backward()
    # external_grad = torch.tensor([1., 1.])
    # Q.backward(gradient=external_grad)
    #
    # print("a.grad:\n", a)
    # print(9 * a ** 2)
    #
    # print(9 * a ** 2 == a.grad)
    # print(-2 * b == b.grad)


    # torch.manual_seed(0)
    #
    # a = torch.rand((3,3), dtype=torch.float32)
    # b = torch.rand((3,3)) + 1
    #
    # c = a + b
    #
    # torch.abs(c)
    # torch.std_mean(c)
    #
    # torch.det(c)
    #
    # U, S, V = torch.svd(c)

    # n_x = 10
    # n_h = 20
    # n_y = 3
    # x = torch.randn((n_x, 1))
    # Wx = torch.randn((n_h, n_x))
    # h = torch.randn((n_h, 1))
    # Wh = torch.randn((n_h, n_h))
    # Wy = torch.randn((n_y, n_h))
    # h = torch.tanh(torch.mm(Wx, x) + torch.mm(Wh, h))
    # y = torch.mm(Wy, h)
    #
    # print(y)

    # data = [[1,2], [3,4]]
    #
    # x_data = torch.tensor(data)
    #
    # np_data = np.array(data)
    # x_data = torch.from_numpy(np_data)
    #
    # x_ones = torch.ones_like(x_data, dtype=torch.int)
    # x_rand = torch.rand_like(x_data, dtype=torch.float)
    #
    # print(x_data)