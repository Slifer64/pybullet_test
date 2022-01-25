import torch
import numpy as np
import torch

if __name__ == '__main__':

    batch_size = 12
    a = torch.randn((batch_size, 4))

    ind = torch.randint(low=0, high=4, size=(batch_size,))

    print(a)

    b = a.gather(1, ind.view(-1, 1))

    c = torch.tensor([a[i, ind[i]].item() for i in range(a.size(0))])

    d = torch.cat((b, c.view(-1,1)), dim=1)

    print(b.shape)
    print(c.shape)

    # print(b.view(1, -1))
    # print(c.view(1, -1))

    print(d)
