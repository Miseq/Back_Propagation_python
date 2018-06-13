import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


# from tensorflow.examples.tutorials.mnist import input_data

def sigma(x):
    return tf.div(tf.constant(1, 0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


def sigma_prime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


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


a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
a_0, y, x_test, y_test = load_cancer_data()
dataset = pd.read_excel("http://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls",
                        sheet_name=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# dataset.to_csv
first = 40
second = 20

w_1 = tf.Variable(tf.truncated_normal([784, first]))
b_1 = tf.Variable(tf.truncated_normal([1, first]))
w_2 = tf.Variable(tf.truncated_normal([first, second]))
b_2 = tf.Variable(tf.truncated_normal([1, second]))
w_3 = tf.Variable(tf.truncated_normal([second, 10]))
b_3 = tf.Variable(tf.truncated_normal([1, 10]))

z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(w_1, w_2), b_2)
a_2 = sigma(z_2)
z_3 = tf.add(tf.matmul(w_2, w_3), b_3)
a_3 = sigma(z_3)

difference = tf.subtract(a_3, y)

# wsteczna propagacja
d_z_3 = tf.multiply(difference, sigma_prime(z_3))
d_bias_3 = d_z_3
d_w_3 = tf.matmul(tf.transpose(a_2), d_z_3)

d_a_2 = tf.matmul(d_z_3, tf.transpose(w_3))
d_z_2 = tf.multiply(d_z_3, sigma_prime(z_2))
d_bias_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmult(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_z_3, sigma_prime(z_1))
d_bias_1 = d_z_1
d_w_1 = tf.matmul(tf.tanspose(a_0), d_z_1)

# aktualizowanie sieci
eta = tf.constant(0.01)
step = [
    tf.assign(w_1, tf.subtract(w_1, tf.multiply(eta, d_w_1))),
    tf.assign(b_1, tf.subtract(b_1, tf.multiply(eta, tf.reduce_mean(d_bias_1, axis=[0])))),
    tf.assign(w_2, tf.subtract(w_2, tf.multiply(eta, d_w_2))),
    tf.assign(b_2, tf.subtract(b_2, tf.multiply(eta, tf.reduce_mean(d_bias_2, axis=[0])))),
    tf.assign(w_3, tf.subtract(w_3, tf.multiply(eta, d_w_1))),
    tf.assign(b_3, tf.subtract(b_3, tf.multiply(eta, tf.reduce_mean(d_bias_3, axis=[0])))),
]

acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sees = tf.InteractiveSession()
sees.run(tf.global_variables_initializer())

epochs = 1000
for iteration in range(epochs):
    batch_xs, batch_ys = iris.train.next_batch(10)
    sees.run(step, feed_dict={a_0: batch_xs, y: batch_ys})

    if i % 1000 == 0:
        res = sees.run(acct_res, feed_dict=
        {a_0: iris.test.images[:1000],
         y: iris.test.labels[:1000]})
        print
        res
