import numpy as np
import typing
import pickle
import matplotlib.pyplot as plt


def relu(z):

    a = z.copy()
    a[a < 0] = 0.
    return a


def relu_dot(z):

    a_dot = np.zeros(z.shape)
    a_dot[z > 0] = 1.
    return a_dot


def sigmoid(z):

    a = 1.0 / (1 + np.exp(-1.0*z))
    return a


def sigmoid_dot(z):

    a = sigmoid(z)
    return a*(1-a)


class DeepNet:

    def __init__(self, layer_dims, hidden_activ_fun='relu', output_activ_fun='sigmoid'):
        self.n_layers = len(layer_dims)  # including the 0-th (input) layer

        # initialize weights
        self.W = [np.array([]) for _ in range(self.n_layers)]
        self.b = [np.array([]) for _ in range(self.n_layers)]
        for l in range(1, self.n_layers):
            self.W[l] = np.random.randn(layer_dims[l], layer_dims[l - 1])*0.01
            self.b[l] = np.zeros((layer_dims[l], 1))

        # cache
        self.Z_ = [np.array([]) for _ in range(self.n_layers)]
        self.A_ = [np.array([]) for _ in range(self.n_layers)]

        # weights diff
        self.dW = [np.array([]) for _ in range(self.n_layers)]
        self.db = [np.array([]) for _ in range(self.n_layers)]

        if hidden_activ_fun == 'relu':
            self.hidden_activation_fun = relu
            self.hidden_activation_fun_dot = relu_dot
        elif hidden_activ_fun == 'sigmoid':
            self.hidden_activation_fun = sigmoid
            self.hidden_activation_fun_dot = sigmoid_dot

        if output_activ_fun == 'relu':
            self.out_activation_fun = relu
            self.out_activation_fun_dot = relu_dot
        elif output_activ_fun == 'sigmoid':
            self.out_activation_fun = sigmoid
            self.out_activation_fun_dot = sigmoid_dot

    def __call__(self, x):
        return self.forward(x)

    def loss(self, y, y_hat):
        m = y.shape[1]
        loss_ = -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat), axis=1) / m
        self.__backward(y, y_hat)
        return np.squeeze(loss_)

    def forward(self, x):
        L = self.n_layers - 1
        A_prev = x
        for l in range(1, L):
            self.A_[l - 1] = A_prev
            Z = np.dot(self.W[l], A_prev) + self.b[l]
            self.Z_[l] = Z
            A_prev = self.hidden_activation_fun(Z)

        self.A_[L-1] = A_prev
        Z = np.dot(self.W[L], A_prev) + self.b[L]
        self.Z_[L] = Z
        A_L = self.out_activation_fun(Z)
        y_hat = A_L
        return y_hat

    def __backward(self, y, y_hat):

        m = y.shape[1]

        L = self.n_layers - 1
        a_L = y_hat
        dA_L = -np.divide(y, a_L) + np.divide(1 - y, 1 - a_L)
        dZ_L = dA_L * self.out_activation_fun_dot(self.Z_[L])
        self.dW[L] = np.dot(dZ_L, self.A_[L-1].T) / m
        self.db[L] = np.sum(dZ_L, axis=1, keepdims=True) / m
        dA_l = np.dot(self.W[L].T, dZ_L)

        for l in range(L-1, 0, -1):
            dZ_l = dA_l * self.hidden_activation_fun_dot(self.Z_[l])
            self.dW[l] = np.dot(dZ_l, self.A_[l - 1].T) / m
            self.db[l] = np.sum(dZ_l, axis=1, keepdims=True) / m
            dA_l = np.dot(self.W[l].T, dZ_l)

    def optimize_step(self, learning_rate=1e-3):

        # update weights
        for l in range(1, self.n_layers):
            self.W[l] -= learning_rate * self.dW[l]
            self.b[l] -= learning_rate * self.db[l]


def load_data():
  X_train = pickle.load(open('data/train_data_X.pkl', 'rb'))
  y_train = pickle.load(open('data/train_data_Y.pkl', 'rb'))

  X_test = pickle.load(open('data/test_data_X.pkl', 'rb'))
  y_test = pickle.load(open('data/test_data_Y.pkl', 'rb'))

  return X_train, y_train, X_test, y_test


def plot_data(X_data, y_data, ind=[]):

    if not ind:
        ind = np.arange(X_data.shape[0])

    y_data = y_data.reshape(-1)

    for i in ind:
        plt.imshow(X_data[i])
        if y_data[i] == 1:
            plt.title('# %d: Cat' % i)
        else:
            plt.title('# %d' % i)
        plt.show()
        plt.clf()


if __name__ == '__main__':

    np.random.seed(0)

    X_train, y_train, X_test, y_test = load_data()

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)

    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)

    # plot_data(X_train, y_train, [0, 10, 50, 100])

    m_train = X_train.shape[0]
    m_test = X_test.shape[0]

    # Reshape the training and test examples
    train_x_flat = X_train.reshape(m_train, -1).T
    test_y_flat = X_test.reshape(m_test, -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flat / 255.
    test_x = test_y_flat / 255.

    n_x = train_x.shape[0]
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    learning_rate = 0.0075

    net = DeepNet(layers_dims)

    loss_data = []

    for epoch in range(200):

        y_hat = net(train_x)
        loss = net.loss(y_train, y_hat)
        net.optimize_step(learning_rate=learning_rate)

        loss_data.append(loss)

    epochs = np.arange(len(loss_data))

    fig, ax = plt.subplots()
    ax.plot(epochs, np.array(loss_data), color='red', lw=2)
    ax.set_ylabel('cost', fontsize=16)
    ax.set_xlabel('iterations', fontsize=16)
    ax.set_title("Learning rate = " + str(learning_rate), fontsize=16)
    plt.show()

