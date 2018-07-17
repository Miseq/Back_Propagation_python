import sys

import numpy as np
from scipy.special import expit


# TODO - comments and documentation....someday

class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_first_hidden, n_second_hidden, epochs, eta,
                 shuffle=False, minibatches=1):
        self.n_output = n_output
        self.n_features = n_features
        self.n_first_hidden = n_first_hidden
        self.n_second_hidden = n_second_hidden
        self.w1, self.w2, self.w3 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.sse = []

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for index, val in enumerate(y):
            onehot[val, index] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_first_hidden * (self.n_features))
        w1 = w1.reshape(self.n_first_hidden, self.n_features)

        w2 = np.random.uniform(-1.0, 1.0, size=self.n_second_hidden * (self.n_first_hidden + 1))
        w2 = w2.reshape(self.n_second_hidden, self.n_first_hidden + 1)

        w3 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_second_hidden + 1))
        w3 = w3.reshape(self.n_output, self.n_second_hidden + 1)
        return w1, w2, w3

    def fit(self, X, y):

        X_data, y_data, = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        change_w1_prev = np.zeros(self.w1.shape)
        change_w2_prev = np.zeros(self.w2.shape)
        change_w3_prev = np.zeros(self.w3.shape)

        for i in range(self.epochs):

            self._show_forward_progress(i)
            if self.shuffle == True:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)

            for idx in mini:
                # forward propagation
                a1, z2, a2, z3, a3, z4, a4 = self._feedforward(X_data[idx], self.w1, self.w2, self.w3)

                # calc gradients
                grad_1, grad_2, grad_3 = self._get_gradient(a1=a1, a2=a2, a3=a3, a4=a4, z2=z2, z3=z3,
                                                            y_enc=y_enc[:, idx],
                                                            w1=self.w1, w2=self.w2, w3=self.w3)

                # updating
                change_w1, change_w2, change_w3 = self.eta * grad_1, self.eta * grad_2, self.eta * grad_3
                self.w1 -= (change_w1 + change_w1_prev)
                self.w2 -= (change_w2 + change_w2_prev)
                self.w3 -= (change_w3 + change_w3_prev)
                change_w1_prev, change_w2_prev, change_w3_prev = change_w1, change_w2, change_w3

        return self

    def predict(self, X):
        a1, z2, a2, z3, a3, z4, a4 = self._feedforward(X, self.w1, self.w2, self.w3)
        y_prediction = np.argmax(z4, axis=0)
        return y_prediction

    def add_bias(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    def _feedforward(self, X, w1, w2, w3):
        a1 = self.add_bias(X, how='column')
        a1 = X
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self.add_bias(a2, how='row')

        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        a3 = self.add_bias(a3, how='row')

        z4 = w3.dot(a3)
        a4 = self._sigmoid(z4)

        return a1, z2, a2, z3, a3, z4, a4

    def _get_gradient(self, a1, a2, a3, a4, z2, z3, y_enc, w1, w2, w3):
        sigma4 = a4 - y_enc
        self.sse.append(np.sum((sigma4[:] / 2) ** 2))
        z3 = self.add_bias(z3, how='row')

        sigma3 = w3.T.dot(sigma4) * self._sigmoid_gradient(z3)
        sigma3 = sigma3[1:, :]
        z2 = self.add_bias(z2, how='row')

        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]

        grad_1 = sigma2.dot(a1)
        grad_2 = sigma3.dot(a2.T)
        grad_3 = sigma4.dot(a3.T)

        return grad_1, grad_2, grad_3

    def _sigmoid(self, z):
        return expit(z)

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _show_forward_progress(self, i):
        sys.stderr.write('\rEpoka: %d/%d' % (i + 1, self.epochs))
        sys.stderr.flush()
