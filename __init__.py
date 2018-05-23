import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sciki import NeuralNetMLP


# Czwartki 15:30 F1
# Niedziela następna godzina 14 F1

def load_cancer_data(test_size=0.0):
    dataset = pd.read_excel("http://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls",
                            sheet_name=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Dane wejściowe:
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, :1]
    le = LabelEncoder()
    y = np.ravel(y)
    y = le.fit_transform(y=y)
    stdc = StandardScaler()  # dla standaryzacji
    mms = MinMaxScaler()

    X = stdc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test


def load_iris_data(test_size=0.0):
    dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    X = dataset.iloc[:, :3]
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


def prepare_neural_net(X_train, kind='breast'):
    if kind == 'cancer':
        nn = NeuralNetMLP(n_output=6, n_features=X_train.shape[1], n_hidden=90,
                          n_hidden_second=45, l2=0.1, l1=0.001, epochs=1000,
                          eta=0.01, alpha=0.001, decrease_const=0.0001, shuffle=True,
                          minibatches=2, random_state=1)

    elif kind == 'mnist':
        nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, n_hidden_second=20, l2=0.1,
                          l1=0.0, epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.0001,
                          shuffle=True, minibatches=50, random_state=1)


    elif kind == 'iris':
        nn = NeuralNetMLP(n_output=3, n_features=X_train.shape[1], n_hidden=30, n_hidden_second=20, l2=0.1, l1=0.1,
                          epochs=800, eta=0.01, alpha=0, decrease_const=0, shuffle=True,
                          minibatches=1, random_state=0)

    else:
        print("Błędny rodzaj sieci!")
        return 0

    return nn


X_train, X_test, y_train, y_test = load_iris_data(test_size=0.1)
# X_train, y_train = load_breast_cancer()

nn = prepare_neural_net(kind='iris', X_train=X_train)

nn.fit(X_train, y_train, print_progress=False)

plt.subplot(2, 1, 1)
plt.plot(range(len(nn.cost_2)), nn.cost_2, color='red')
plt.title('aktualizacje drugiej warstwy')
plt.xlabel('epoki')
plt.ylabel('koszt')

plt.subplot(2, 1, 2)
plt.plot(range(len(nn.cost_1)), nn.cost_1)
plt.title('akutalizacje pierwszej warstwy')
plt.ylabel('epoki')
plt.xlabel('koszt')

# plt.tight_layout()
plt.show()
print(y_train)

y_train_pred = nn.predict(
    X_train)  # TODO uzupełnić ładne wykresiki, dodać wyliczanie dokładności zarówno dla testowej jak i uczącej
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print("dokładność wobec danych uczących: %.2f%%" % (acc * 100))
# print("koszt minimalny: %d" %(np.min(nn.cost_,axis=0)))
# print("koszt średni: %d" %(np.average(nn.cost_, axis=0)))
# print("koszt max: %d" %(np.max(nn.cost_, axis=0)))
# batches = np.array_split(range(len(nn_breast_cancer.cost_)), 1000)
# cost_ary = np.array(nn_breast_cancer.cost_)
# cost_avgs = [np.mean(cost_ary[i]) for i in batches]
# plt.plot(range(len(cost_avgs)), cost_avgs, color = 'red')
# plt.xlabel('Epoki')
# plt.ylabel('Koszt')
# plt.tight_layout()
# plt.show()
