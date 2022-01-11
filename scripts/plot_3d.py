
def plot_3d_line():

    # importing mplot3d toolkits, numpy and matplotlib
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')

    # defining all 3 axes
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z)
    y = z * np.cos(25 * z)

    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('3D line plot geeks for geeks')
    plt.show()


def plot_3d_scatter():
    # importing mplot3d toolkits
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')

    # defining axes
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z)
    y = z * np.cos(25 * z)
    c = x + y
    ax.scatter(x, y, z, c = c)

    # syntax for plotting
    ax.set_title('3d Scatter plot geeks for geeks')
    plt.show()


def plot_surface():
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt

    # function for z axea
    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

    # x and y axis

    # X = np.outer(np.linspace(-1,5,10), np.ones(10))
    # Y = X.copy().T

    x = np.linspace(-1, 5, 10)
    y = np.linspace(-1, 5, 10)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='green')
    ax.set_title('Surface plot geeks for geeks')
    plt.show()


# =========== MAIN ===========
if __name__ == '__main__':

    # plot_3d_line()
    # plot_3d_scatter()
    plot_surface()