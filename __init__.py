import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer

from Model import NeuralNetMLP


# First of all - i am beginner ;)


# Loading breast_tissue_data
def load_cancer_data(test_size=0.0):
    dataset = pd.read_excel(".\BreastTissue.xls",
                            sheet_name=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Dane wejściowe:
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, :1]
    le = LabelEncoder()
    y = np.ravel(y)
    y = le.fit_transform(y=y)
    norm = Normalizer()
    X = norm.fit_transform(X)
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
    norm = Normalizer()
    X = norm.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_cancer_data(test_size=0.0)
acc_in_second_layer = []
acc_in_first_layer = []
avg_acc_in_second = []
S1 = []
S2 = []
sse = []
cost = []

for i in range(1, 30):
    print(f'\nFirst layer\' neurons: {i}')
    for j in range(1, 30):
        nn = NeuralNetMLP(n_output=len(np.unique(y_train)), n_features=X_train.shape[1], n_first_hidden=i,
                          n_second_hidden=j,
                          epochs=30000, eta=0.001, shuffle=False, minibatches=1)
        if j % 5 == 0:
            nn.fit(X_train, y_train)
            y_train_pred = nn.predict(X_train)
            acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
            acc_in_second_layer.append(acc)
            sse.append(np.min(nn.sse))
            S2.append(j), S1.append(i)
            x = range(len(nn.sse))
            y = nn.sse
            print(f'\nSecond layer\' neurons: {j}')

# It's saving data in .xls type for futher analys
wb = xlwt.Workbook()
worksheet = wb.add_sheet("Sheet1")
for i in range(len(acc_in_second_layer)):
    worksheet.write(i, 0, S1[i - 1])
    worksheet.write(i, 1, S2[i - 1])
    worksheet.write(i, 2, acc_in_second_layer[i - 1])
    worksheet.write(i, 3, sse[i - 1])
wb.save('output.xls')

# This piece of code makes a small plot of model's cost
plt.plot(range(len(nn.sse)), nn.sse)
plt.ylabel('Koszt')
plt.xlabel('Epoki')
plt.tight_layout()
plt.show()

# Code below makes a simple plot that shows which records are not correctly predicted
# which class do they belong to and which class was predicted for them
difference = []
true = []
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(y_train)):
    if y_train[i] != y_train_pred[i]:
        difference.append(y_train_pred[i])
        true.append(y_train[i])
    else:
        difference.append(20)
        true.append(20)
for i, j in zip(range(len(y_train)), difference):
    point = i
    ax.annotate(str(point), xy=(i, j + 0.01))
plt.plot(range(len(y_train)), difference, 'r*')
plt.plot(true, 'bo')
ax.set_ylim(0, nn.n_output)
ax.set_xlim(0, 150)
plt.ylabel('class')
plt.xlabel('record')
plt.grid()
plt.show()

# Show acc in console
print("accuracy: %.2f%%" % (acc * 100))
