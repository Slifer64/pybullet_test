import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    a = np.array([[1, 1], [1, 1]])

    w = np.array([[[2, 2], [2, 2]], [[5, 5], [5, 5]], [[8, 8], [8, 8]]])

    w = np.array([[2, 5, 8], [2, 5, 8], [2, 5, 8], [2, 5, 8]])

    n_C, f, _ = w.shape

    print(w.shape)

    w = w.reshape((f, f, n_C))

    print(w)

    exit()

    z = np.sum(a*w, axis=(1, 2))


    print(z)


