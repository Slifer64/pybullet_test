import numpy as np
import matplotlib.pyplot as plt

def applyGlobalPlotChanges():

    # change the fontsize globally
    # You have to set these before the plot() function call since if you try to apply them afterwards,
    # no change will be made.
    plt.rcParams['font.size'] = '16'

    font = {'family': 'normal', 'weight': 'bold', 'size': 22}

    plt.rc('font', **font)

    # set x axis tight
    plt.rcParams['axes.xmargin'] = 0


if __name__ == '__main__':

    np.random.seed(0)

    # execution doesn't halt at plt.show()
    plt.ion()
    # plt.ioff()

    x = np.linspace(-np.pi, np.pi, 1000)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # applyGlobalPlotChanges()

    fig, ax = plt.subplots()

    ax.plot(x, y_sin, color='blue', linestyle='-', linewidth=2, label='sin')
    ax.plot(x, y_cos, color='magenta', linestyle='--', linewidth=2, label='cos')
    ax.set_title('Plot example', fontsize=20)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)

    # Determine the ticks
    x_ticks = x[np.linspace(0, x.size - 1, 10, dtype=int)]
    ax.xaxis.set_ticks(x_ticks) # set the x ticks
    ax.xaxis.set_ticklabels(['{0:.2f}'.format(val) for val in x_ticks]) # set the precision of each tick to 2 decimals

    # axis tight
    ax.autoscale(enable=True, axis='both', tight=True)  # axis='x', 'y'
    # ax.set_xmargin(0)

    # # set specific limits
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-1, 1)





    # x = np.linspace(0, 2, 200)
    # data = []
    # data.append({'x': x, 'y': x, 'label': 'linear', 'linestyle': '-', 'color': 'blue'})
    # data.append({'x': x, 'y': x**2, 'label': 'quadratic', 'linestyle': ':', 'color': 'green'})
    # data.append({'x': x, 'y': x**3, 'label': 'cubic', 'linestyle': '-.', 'color': 'red'})
    #
    # fig, ax = plt.subplots()
    # for dat in data:
    #     ax.plot(dat['x'], dat['y'], label=dat['label'], color=dat['color'], linestyle=dat['linestyle'], linewidth=2)
    # ax.legend()
    # ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
    # ax.grid(True)
    # ax.axis([np.min(x), np.max(x), 0, 2])
    # # ax.set_yscale('log')


    plt.show()

