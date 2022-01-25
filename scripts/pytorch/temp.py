import torch
import numpy as np
import torch

if __name__ == '__main__':

    a = torch.tensor(tuple(i**2 for i in range(10)))


    print(a[[1, 3, 5]])
