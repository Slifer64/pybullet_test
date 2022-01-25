import gym
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    random.seed(0)

    x = torch.linspace(-math.pi, math.pi, 400)
    y = torch.sin(x)
    y_msr = y + 0.15 * torch.randn_like(y)


    filt_w = 17
    n_times = 1
    y_mav = y_msr.clone()
    for _ in range(n_times):
        y_mav = torch.cat((y_mav[0]*torch.ones(int(filt_w/2)), y_msr, y_mav[-1]*torch.ones(int(filt_w/2)))).\
            unfold(0, filt_w, 1).mean(1).flatten(0)

    a_f = 0.1
    y_iir = y_msr.clone()
    for i in range(1, y_iir.numel()):
        y_iir[i] = a_f*y_iir[i] + (1-a_f)*y_iir[i-1]


    fig, ax = plt.subplots()
    ax.plot(x, y_msr, lw=2, color='red', label='y_msr')
    ax.plot(x, y_mav, lw=3, color='green', label='y_mav (' + str(filt_w) + ')')
    ax.plot(x, y_iir, lw=3, color='cyan', label='y_iir (' + str(a_f) + ')')
    ax.plot(x, y, lw=2, color='blue', label='y', linestyle='--')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.legend(fontsize=16)

    plt.show()