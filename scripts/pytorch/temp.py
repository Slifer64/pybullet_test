import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def loss_fun(x, y):

    return x.sum().item(), y.item()

if __name__ == '__main__':

    torch.random.manual_seed(0)

    x_data = torch.randn((5, 20))
    y_data = torch.randn(5)

    data_set = torch.utils.data.TensorDataset(x_data, y_data)

    a = [loss_fun(x, y) for x, y in data_set]

    print(*a)

