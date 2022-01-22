import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def dummy_fun(x):
     y = 0
     xi = x.clone()
     for i in range(len(x)):
         y += xi
         xi *= xi

     return y

if __name__ == '__main__':

    # trainInTorch()

    torch.random.manual_seed(0)

    a0 = torch.nn.Parameter(torch.tensor([0.]))
    param = torch.nn.Parameter(torch.tensor([3., 2., 1.]))

    print(param.size())

    # x = torch.linspace(0, 2, 10)
    #
    # y = a0
    # for i in range()
    #
    # print(param)
    # print(param[:, 1])
    # print(param.data[:, 1])