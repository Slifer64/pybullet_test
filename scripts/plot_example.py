import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':


    np.random.seed(0)

    Tf = 2
    n_data = 100

    x = np.linspace(0,Tf, n_data)

    y1 = np.sin(x)
    y2 = np.power( np.cos(x) , 2)

    y = np.row_stack((y1, y2))

    n_rows, n_cols = y.shape

    print("y: ", n_rows, " x ", n_cols)

    fig, axs = plt.subplots(2,1)

    y_labels = ['y1', 'y2']

    # for i,ax in enumerate(axs):
    #     ax.plot(x, y[i,:], linewidth=2, linestyle='-.', color='red', label=y_labels[i])
    #     if i == len(axs)-1:
    #         ax.set_xlabel("time [s]")
    #     ax.set_ylabel(y_labels[i])
    #     ax.legend()

    # plt.show()

    data = []
    data.append( {'x': x, 'y': x, 'label': 'linear', 'linestyle': '-', 'color': 'blue'} )
    data.append({'x': x, 'y': x**2, 'label': 'quadratic', 'linestyle': ':', 'color': 'green'})
    data.append({'x': x, 'y': x**3, 'label': 'cubic', 'linestyle': '-.', 'color': 'red'})

    fig, ax = plt.subplots()
    for dat in data:
        ax.plot(dat['x'], dat['y'], label=dat['label'], color=dat['color'], linestyle=dat['linestyle'], linewidth=2)
    ax.legend()
    ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
    ax.grid(True)
    ax.axis([np.min(x), np.max(x), 0, 2])
    # ax.set_yscale('log')

    plt.show()

