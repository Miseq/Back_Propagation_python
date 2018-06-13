import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sciki import NeuralNetMLP


def load_cancer_data(test_size=0.0):
    dataset = pd.read_excel("C:/Users\krzyc\Downloads\BreastTissue (1).xls",
                            sheet_name=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Dane wejściowe:
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, :1]
    le = LabelEncoder()
    y = np.ravel(y)
    y = le.fit_transform(y=y)
    stdc = StandardScaler()  # dla standaryzacji
    X = stdc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test


def load_iris_data(test_size=0.0):
    dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    dataset = dataset.replace(r'\\n', '', regex=True)
    X = dataset.iloc[:, :4]
    y = dataset.iloc[:, 4]
    le = LabelEncoder()
    y = np.ravel(y)
    y = le.fit_transform(y=y)
    stdc = StandardScaler()
    X = stdc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test


def prepare_Mnist_data(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def load_mnist_data():
    X_train, Y_train = prepare_Mnist_data('MNIST', kind='train')
    X_test, Y_test = prepare_Mnist_data('MNIST', kind='t10k')
    return X_train, X_test, Y_train, Y_test


def prepare_neural_net(X_train, kind, neurons_first_layer, neurons_second_layer):
    if kind == 'cancer':
        nn = NeuralNetMLP(n_output=6, n_features=X_train.shape[1], n_hidden=neurons_first_layer,
                          n_hidden_second=neurons_second_layer, l2=0.1, l1=0.001, epochs=1000,
                          eta=0.01, alpha=0, decrease_const=0, shuffle=True,
                          minibatches=2, random_state=1)

    elif kind == 'mnist':
        nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=neurons_first_layer,
                          n_hidden_second=neurons_second_layer,
                          l2=0.1, l1=0.0, epochs=800, eta=0.001, alpha=0.001, decrease_const=0.0001,
                          shuffle=True, minibatches=3, random_state=1)


    elif kind == 'iris':
        nn = NeuralNetMLP(n_output=3, n_features=X_train.shape[1], n_hidden=neurons_first_layer,
                          n_hidden_second=neurons_second_layer,
                          l2=0.1, l1=0.1,
                          epochs=800, eta=0.01, alpha=0, decrease_const=0, shuffle=True,
                          minibatches=1, random_state=0)

    else:
        print("Błędny rodzaj sieci!")
        return 0

    return nn


X_train, X_test, y_train, y_test = load_mnist_data()
second_layer = [i for i in range(0, 30)]
zs = []
xs = []
ys = []
yw = []
# for i in range(0, 30):
#     nn = NeuralNetMLP(n_output=3, n_features=X_train.shape[1], n_hidden=50, n_hidden_second=second_layer[i], l2=0.1,
#                       l1=0.1,
#                       epochs=800, eta=0.01, alpha=0, decrease_const=0, shuffle=True,
#                       minibatches=1, random_state=0)
#     nn.fit(X_train, y_train)
#     y_train_pred = nn.predict(X_train)
#     acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
#     print(i)
#     ys.append([acc, i])
#     xs.append([acc, 30])
#     zs.append([acc])
#     yw.append(acc)
nn = prepare_neural_net(X_train, 'mnist', neurons_first_layer=50, neurons_second_layer=30)
nn.fit(X_train, y_train)
y_train_pred = nn.predict(X_test)
acc = np.sum(y_test == y_train_pred, axis=0) / X_test.shape[0]
# ys.append([acc, i])
# xs.append([acc, 30])
# zs.append([acc])
# yw.append(acc)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# z = np.array(zs)
# y = np.array(ys)
# x = np.array(xs)
# ax.set_xlabel('pierwsza warstwa neuronów')
# ax.set_ylabel('druga wastwa neuronów')
# ax.set_zlabel('dokładność')
# ax.plot_surface(x, y, z )
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.xlabel('ilość neuronów 2 warstwy ukrytej')
# plt.ylabel('dokładność przewidywania w %')
# plt.plot(range(30), yw)
# for i,j in zip(range(30),yw):
#     point = round(j,3)
#     ax.annotate(str(point), xy=(i,j+0.01))
#     ax.annotate(str(i), xy=(i, j - 0.02))
# plt.show()

train_label = plt.plot(y_train, 'b*')
train_pred_label = plt.plot(y_train_pred, 'ro')
plt.xlabel('ilość próbek')
plt.ylabel('klasa wyjściowa')
plt.tight_layout()
plt.grid()
plt.show()

difference = []
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(y_train)):
    if y_train[i] != y_train_pred[i]:
        difference.append(y_train[i])
    else:
        difference.append(20)
ax.set_ylim(0, 2)
plt.plot(range(len(y_train)), difference, 'r*')
for i, j in zip(range(len(y_train)), difference):
    ax.annotate(str(i), xy=(i, j + 0.15))
plt.ylabel('Klasa')
plt.xlabel('próbki')
plt.grid()
plt.show()

batches = np.array_split(range(len(nn.cost_total)), 1000)
cost_array = np.array(nn.cost_total)
cost_avgs = [np.mean(cost_array[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='green')
plt.ylabel('Koszt')
plt.xlabel('Epoki x ilość podzbiorów')
plt.tight_layout()
plt.show()


print("dokładność wobec danych uczących: %.2f%%" % (acc * 100))
