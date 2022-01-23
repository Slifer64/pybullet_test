import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def display_digit(digit_vec: torch.Tensor):

    plt.imshow(digit_vec.numpy().reshape((28, 28)), cmap="gray")
    plt.show()


def downloadData():
    from pathlib import Path
    import requests
    import pickle
    import gzip

    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        return x_train, y_train, x_valid, y_valid


class MyDataLoader:

    def __init__(self, x_data, y_data, batch_size, shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.bs = batch_size
        self.n_data = x_data.size(0)
        self.ind = []
        self.shuffle(shuffle)

    def setBatchSize(self, batch_size):
        self.bs = batch_size

    def shuffle(self, set):
        if set:
            self.ind = np.random.permutation(self.n_data)
        else:
            self.ind = np.arange(self.n_data)

    def __getitem__(self, item):
        return self.x_data[item, :], self.y_data[item]

    def __len__(self):
        return self.n_data

    def __iter__(self):
        self.i1 = 0
        self.i2 = self.bs
        return self

    def __next__(self):
        if self.i2 == self.n_data:
            raise StopIteration
        else:
            self.i1 = self.i2
            self.i2 += self.bs
            if self.i2 > self.n_data:
                self.i2 = self.n_data
            return self.__getitem__(self.ind[self.i1:self.i2])


class Mnist_model:

    def __init__(self):

        self.W = torch.randn((784, 10)) / math.sqrt(784)
        self.W.requires_grad_()
        self.b = torch.zeros(10)
        self.b.requires_grad_()

        # or
        # self.W = torch.randn((784, 10), requires_grad=True)
        # with torch.no_grad():
        #     self.W /= math.sqrt(784)

    def forward(self, x: torch.Tensor):
        return x @ self.W + self.b

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def parameters(self):
        yield self.W
        yield self.b

    def train(self):
        pass

    def eval(self):
        pass


class Mnist_net(torch.nn.Module):

    def __init__(self):
        super(Mnist_net, self).__init__()
        # self.W = torch.nn.Parameter(torch.randn((784, 10)) / math.sqrt(784))
        # self.b = torch.nn.Parameter(torch.zeros(10))
        self.lin = torch.nn.Linear(784, 10)

    def forward(self, x: torch.Tensor):
        # return x @ self.W + self.b
        return self.lin(x)


class Transform_layer(torch.nn.Module):
    def __init__(self, transform_fun):
        super().__init__()
        self.tf_fun = transform_fun

    def forward(self, input):
        return self.tf_fun(input)


class Mnist_CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        input_tf = Transform_layer(lambda x: x.view(-1, 1, 28, 28))  # batch_size -1 to assign it automatically
        conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        pool = torch.nn.AvgPool2d(2)
        flatten = Transform_layer(lambda x: x.view(x.size(0), -1))

        # create dummy input to calculate output dim so far:
        with torch.no_grad():
            x = torch.randn(28**2)
            x = input_tf(x)
            x = F.relu(conv1(x))
            x = F.relu(conv2(x))
            x = pool(x)
            x = flatten(x)  # torch.flatten(x, 1)  # don't flatten batch_size (0)
            n_conv_out = x.numel()

        out_layer = torch.nn.Linear(n_conv_out, 10)

        self.net = torch.nn.Sequential(
            input_tf,
            conv1, torch.nn.ReLU(),
            conv2, torch.nn.ReLU(),
            pool, flatten,  #torch.nn.Flatten(1),  # don't flatten batch_size (0)
            out_layer
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.to(device))


def calc_accuracy(model, data_loader: MyDataLoader):

    # i1, i2 = 0, batch_size
    total = 0.
    correct = 0.
    model.eval()
    with torch.no_grad():
        # for i in range(c_data.numel() // batch_size):
        for xb, cb in data_loader:

            # xb = x_data[i1:i2, :]
            # cb = c_data[i1:i2]

            y_pred = model(xb)

            cb_hat = torch.argmax(y_pred, dim=1)

            total += cb.numel()
            correct += (cb.to(device) == cb_hat).sum().item()

    return 100.*correct/total


def trainModel(train_loader, model, loss_fun, optimizer, valid_loader=None):
    model.train()
    for epoch in range(2):

        # i1, i2 = 0, batch_size
        # for i in range(n_train // batch_size):
        for i, (xb, cb) in enumerate(train_loader):

            # xb = x_train[i1:i2, :]
            # cb = y_train[i1:i2]

            # i1 += batch_size
            # i2 += batch_size

            y_pred = model(xb)

            loss = loss_fun(y_pred.to(device), cb.to(device))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 100 == 99:
                print('[%d, %4d]: loss = %.2f' % (epoch + 1, i + 1, loss.item()))

        if valid_loader is not None:
            with torch.no_grad():
                valid_loss = sum(loss_fun(model(xb), yb.to(device)).item() for xb, yb in valid_loader) / len(valid_loader)
            print('epoch %d: validation loss = %.3f' % (epoch+1, valid_loss))



if __name__ == '__main__':

    torch.random.manual_seed(0)

    x_train, y_train, x_valid, y_valid = downloadData()
    x_train, y_train, x_valid, y_valid = map(torch.as_tensor, (x_train, y_train, x_valid, y_valid))
    # n_train, c = x_train.shape

    # train_loader = MyDataLoader(x_train, y_train, batch_size=64, shuffle=True)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=64, shuffle=True)

    valid_loader = MyDataLoader(x_valid, y_valid, batch_size=128, shuffle=False)
    # valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid),
    #                                            batch_size=200, shuffle=False)

    # valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
    # valid_dataset[5:10] --> x_valid[5:10, :], y_valid[5:10]
    # However, implementing __getitem__() explicitly is more concrete

    # display_digit(x_train[0, :])

    # model = Mnist_net()
    # model = Mnist_model()
    model = Mnist_CNN()
    model.to(device)
    # loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_fun = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    trainModel(train_loader, model, loss_fun, optimizer, valid_loader=valid_loader)

    # train_loader.setBatchSize(200)
    # train_loader.shuffle(False)
    acc = calc_accuracy(model, train_loader)
    print('Training accuracy: %.3f %%' % acc)

    acc = calc_accuracy(model, valid_loader)
    print('Validation accuracy: %.3f %%' % acc)



    # plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    # plt.show()
    # print(x_train.shape)
