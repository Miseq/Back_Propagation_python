import numpy as np
from scipy.special import expit


class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30, n_hidden_second=15, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0,
                 decrease_const=0.0,
                 shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_hidden_second = n_hidden_second
        self.l1 = l1
        self.l2 = l2
        self.w1, self.w2, self.w3 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

        # X.shape[1] czyli dla X=[106][9] X.shape[0]=106 zaś X.shape[1]=9
        # eta - wspłczynnik uczenia
        # l1 i l2 - wyrażenia regularyzacji patrz logistyczna funkcja kosztu  '_get_cost'

        # alpha parametr dla uczenia momentowego, przyśpisza uczenia poprzez określenie częsci wartości gradientu dodawanej do
        #       zaktualizwanych wag, dla tego zbioru tego nie widać ale dla zbioru MNIST(60k) próbek przyśpieszał uczenie

        # decrease_const(ang. stała redukcji) element adaptacyjnego współczynnika uczenia n, malejącego z upływem czasu(epokami)
        #                                    n/1+txd, patrz metoda 'fit'

        # shuffle - tasowanie w celu zapobieżenie cykliczności, na obecną chwilę poprawia dopasowanie o ok 5%
        # minibatches - dzieli dane uczące na k podzbiorów w każdej epoce, dla każdego podzbioru gradient obliczany jest
        #               odddzielnie

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for index, val in enumerate(y):
            onehot[val, index] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)

        w2 = np.random.uniform(-1.0, 1.0, size=self.n_hidden_second * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_hidden_second, self.n_hidden + 1)

        w3 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden_second + 1))
        w3 = w3.reshape(self.n_output, self.n_hidden_second + 1)
        return w1, w2, w3

    def _sigmoid(self, z):
        # wartosc expit wynosi 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('Atrybut "how" musi mieć wartość "cloumn" albo "row"')
        return X_new

    def _feedforward(self, X, w1, w2, w3):
        a1 = self._add_bias_unit(X, how='column')  # pobudzenie warstwy weściowej

        z2 = w1.dot(a1.T)  # aktywacja warstwy wejściowej i w dół analogicznie
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')

        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        a3 = self._add_bias_unit(a3, how='row')

        z4 = w3.dot(a3)
        a4 = self._sigmoid(z4)

        return a1, z2, a2, z3, a3, z4, a4

    def _L2_reg(self, lambda_, w1, w2, w3):
        return (lambda_ / 2.0) * (
            np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:]).sum() + np.sum(
                w3[:, 1:]).sum())  # TODO obczaic czy nie ma błędów

    def _L1_reg(self, lambda_, w1, w2, w3):
        return (lambda_ / 2.0) * (
            np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum() + np.abs(
                w3[:, 1:]).sum())  # TODO obczaic czy nie ma błędów

    def _get_cost(self, y_enc, output, w1, w2, w3):
        # logistyczna funkcja kosztu
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2, w3)
        L2_term = self._L2_reg(self.l2, w1, w2, w3)
        cost_total = cost + L1_term + L2_term
        cost_1 = cost / 2 + L1_term
        cost_2 = cost / 2 + L2_term
        return cost_total, cost_1, cost_2

    def _get_gradient(self, a1, a2, a3, a4, z2, z3, y_enc, w1, w2, w3):
        # propagacja wsteczna
        sigma4 = a4 - y_enc
        z3 = self._add_bias_unit(z3, how='row')

        sigma3 = w3.T.dot(sigma4) * self._sigmoid_gradient(z3)
        sigma3 = sigma3[1:, :]
        z2 = self._add_bias_unit(z2, how='row')

        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]

        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        grad3 = sigma4.dot(a3.T)

        # regularyzacja
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 + np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])
        grad3[:, 1:] += self.l2 * w3[:, 1:]
        grad3[:, 1:] += self.l1 * np.sign(w3[:, 1:])

        return grad1, grad2, grad3

    def predict(self, X):
        a1, z2, a2, z3, a3, z4, a4 = self._feedforward(X, self.w1, self.w2, self.w3)
        y_pred = np.argmax(z4, axis=0)
        print(y_pred)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_total = []
        self.cost_1 = []
        self.cost_2 = []
        X_data, y_data, = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        delta_w3_prev = np.zeros(self.w3.shape)

        for i in range(self.epochs):

            # współczynnik uczenia adaptacyjnego
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                print('\rEpoka: %d/%d ' % (i + 1, self.epochs))

                # wypisuje ilość epok, obecna/maks

            if self.shuffle == True:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # propagacja jednokierunkowa
                a1, z2, a2, z3, a3, z4, a4 = self._feedforward(X_data[idx], self.w1, self.w2, self.w3)
                cost_tot, cost_L1, cost_L2 = self._get_cost(y_enc=y_enc[:, idx], output=a4, w1=self.w1, w2=self.w2,
                                                            w3=self.w3)
                self.cost_total.append(cost_tot)
                self.cost_1.append(cost_L1)
                self.cost_2.append(cost_L2)

                # oblicza gradient za pomocą wstecznej propagacji
                grad1, grad2, grad3 = self._get_gradient(a1=a1, a2=a2, a3=a3, a4=a4, z2=z2, z3=z3, y_enc=y_enc[:, idx],
                                                         w1=self.w1,
                                                         w2=self.w2, w3=self.w3)

                # aktualizacja wagi
                delta_w1, delta_w2, delta_w3 = self.eta * grad1, self.eta * grad2, self.eta * grad3
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                self.w3 -= (delta_w3 + (self.alpha * delta_w3_prev))
                delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3

        return self
