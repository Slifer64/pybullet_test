import numpy as np
import pickle
import matplotlib.pyplot as plt
import os





if __name__ == '__main__':

  X_train = pickle.load(open('train_data_X.pkl', 'rb'))
  y_train = pickle.load(open('train_data_Y.pkl', 'rb'))

  X_test = pickle.load(open('test_data_X.pkl', 'rb'))
  y_test = pickle.load(open('test_data_Y.pkl', 'rb'))

  print('X_train: ', X_train.shape)
  print('y_train: ', y_train.shape)

  print('X_test: ', X_test.shape)
  print('y_test: ', y_test.shape)

  m_train = X_train.shape[0]
  for i in range(m_train):
    plt.imshow(X_train[i])
    if y_train[0, i] == 1:
      plt.title('Cat')
    plt.show()
    plt.clf()

  #   a = np.array([0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,
  # 0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,
  # 0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,1,
  # 1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,
  # 1,1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,
  # 0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0])
  #
  #   a = a.reshape(1, -1)
  #
  #   print('a: ', a.shape)
  #
  #   pickle.dump(a, open('train_data_Y.pkl', 'wb'))
  #   exit()
  #
  #   X = pickle.load(open('train_data_X.pkl', 'rb'))
  #
  #   print('X before: ', X.shape)
  #
  #   X = np.concatenate((X, a), axis=0)
  #   pickle.dump(X, open('train_data_X.pkl', 'wb'))
  #
  #   print('X after: ', X.shape)

