import math
import numpy as np
import torch

def choose_mat_cols_from_index_vec():

    batch_size = 12
    n_cols = 4

    a = torch.randn((batch_size, n_cols))
    ind = torch.randint(low=0, high=n_cols, size=(batch_size,))

    b = a.gather(1, ind.view(-1, 1)).flatten()
    c = torch.tensor([a[i, ind[i]].item() for i in range(a.size(0))])

    for i in range(c.size(0)):
        assert b[i] == c[i]

    # align them like this [b c]
    d = torch.cat((b.view(-1, 1), c.view(-1, 1)), dim=1)
    d = torch.cat((b.unsqueeze(1), c.unsqueeze(1)), dim=1)


if __name__ == '__main__':

    choose_mat_cols_from_index_vec()

    # # ======== General ===========
    # a = np.array([1, 2, 3])
    # t = torch.from_numpy(a)  # shares the same memory as a
    # t[0] = -1  # changes a as well
    # assert a[0] == -1
    #
    # a = np.array([1, 2, 3])
    # t = torch.tensor(a)  # copies the data from a
    # t2 = torch.as_tensor(t)  # shares the same memory as t
    # t2[0] = -1
    # assert a[0] == 1
    # assert t[0] == t2[0]
    #
    # t = torch.tensor([1, 2, 3])
    # t2 = torch.tensor(t)  # copies the data from t
    # t2[0] = -1
    # assert t[0] == 1
    #
    # # torch.zeros/ones/empty(size_tuple)
    # # torch.zeros/ones/empty_like(another_tensor)
    # # torch.full(size_tuple, value)
    # # torch.full_like(another_tensor, value)
    #
    # # torch.arange(start=0, end=10, step=2) == tensor([0, 2, 4, 6, 8])
    # # torch.linspace(0, 1, 100)
    #
    # print(t.numel())

    # ======== Indexing, Slicing, Joining, Mutating ===========

    # # ======== Random ===========
    # torch.random.manual_seed(0)
    # # rand, rand_like
    # # radint, randint_like
    # # randn , randn_like
    # torch.randperm(10) # rand perm of 0...9
    #
    # # ======= Autograd ========
    # x = torch.zeros(1, requires_grad=True)
    # with torch.no_grad():
    #     y = x * 2
    # assert y.requires_grad == False
    #
    # with torch.set_grad_enabled(False):
    #     y = x * 2
    # assert y.requires_grad == False
    #
    # torch.set_grad_enabled(True)  # this can also be used as a function
    # y = x * 2
    # assert y.requires_grad == True
    #
    # t = torch.tensor([2, 3, 4, 5], dtype=torch.float)
    # r_t = torch.ones_like(t) / t
    # print(r_t, '\n', torch.reciprocal(t))