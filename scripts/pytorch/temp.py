import math
import numpy as np
import torch
import matplotlib.pyplot as plt



if __name__ == '__main__':

    torch.random.manual_seed(0)

    x = torch.randn((3, 1, 2, 2))

    print(x)

    print(torch.flatten(x, 1))

