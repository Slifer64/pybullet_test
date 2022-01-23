import math
import numpy as np
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    torch.random.manual_seed(0)

    n_out = 10
    y = torch.randn(n_out)

    f_y = y - torch.log(torch.sum(torch.exp(y))) * torch.ones_like(y)

    f_y2 = y - y.exp().sum().log() * torch.ones_like(y)

    print(f_y)
    print(f_y2)

    torch.nn.CrossEntropyLoss