import torch
import numpy as np
import torch

if __name__ == '__main__':

    a = tuple(i**2 for i in range(10))

    x = slice(2, 5)

    print(type(x))

    a_ = [[i, i**2, i**3] for i in range(10)]

    a = torch.tensor(a_)
    m, n = a.shape

    ind = torch.tensor([i%n for i in range(m)])

    print(ind)
    print(ind.shape)

    print(a)
    print(a.shape)

    b = torch.gather(a, 1, ind.view(-1, 1)).view()

    print(b)

