#Imports
import pandas as pd
import numpy as np
from numpy.linalg import pinv, inv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Functions
def to_bipolar(y, labels2index_map):
    y_bipolar = np.ones((len(y), len(labels2index_map))) * -1
    for i, label in enumerate(y):
        y_bipolar[i, label] = 1
    return y_bipolar

def sigmoid(data):
    result = 1 / (1 + np.exp(-1 * data))
    return result

#Main
if __name__ == '__main__':
    print('---ONLINE ELM - MULTI-CLASS---')
    fname = 'iris.csv'
    print('Loading data from file: {}'.format(fname))
    data = pd.read_csv(fname).values
    np.random.shuffle(data)
    print('{} samples loaded with {} features'.format(data.shape[0], data.shape[1] - 1))

    feats = data[:, :-1]
    labels_raw = data[:, -1]
    labels_set = set(labels_raw)
    print('Labels are: {}'.format(labels_set))
    labels2index_map = {label: ind for ind, label in enumerate(labels_set)}
    print('Labels to index map: {}'.format(labels2index_map))
    labels = np.array([labels2index_map[label] for label in labels_raw])

    scaler = MinMaxScaler()
    feats = scaler.fit_transform(feats)
    print('Scaling Features Done')

    train_X, test_X, train_y, test_y = train_test_split(feats, labels, test_size=0.1, random_state=42)
    print('Divided into training and testing set')

    train_y_bipolar = to_bipolar(train_y, labels2index_map)
    test_y_bipolar = to_bipolar(test_y, labels2index_map)
    print('Converted labels to bipolar')

    nHidden = 10
    N0 = 50

    input_dim = train_X.shape[1]
    hidden_dim = nHidden
    output_dim = train_y_bipolar.shape[1]
    i2h = np.random.uniform(-1, 1, (input_dim, hidden_dim))

    #ELM Training
    ##Initial Block
    X0 = train_X[:N0]
    y0 = train_y_bipolar[:N0]
    H0 = sigmoid(np.dot(X0, i2h))
    M0 = inv(np.dot(np.transpose(H0), H0))
    beta0 = np.dot(np.dot(M0, np.transpose(H0)), y0)
    H = H0
    M = M0
    beta = beta0
    ##Subsequent Block
    for X, y in zip(train_X[N0:], train_y_bipolar[:N0]):
        X = X.reshape((1, 4))
        y = y.reshape((1, 3))
        H = sigmoid(np.dot(X, i2h))
        # print('H', H.shape)
        Dr = np.eye(H.shape[0]) + np.dot(np.dot(H, M), np.transpose(H))
        Nr1 = np.dot(M, np.transpose(H))
        Nr2 = inv(Dr)
        Nr = np.dot(np.dot(np.dot(Nr1, Nr2), H), M)
        M = M - Nr
        # print('M', M.shape)
        Nr3 = y - np.dot(H, beta)
        Nr4 = np.dot(np.dot(M, np.transpose(H)), Nr3)
        beta = beta + Nr4

    # H = sigmoid(np.dot(train_X, i2h))
    # beta = np.dot(pinv(H), train_y_bipolar)

    #ELM Testing
    H = sigmoid(np.dot(test_X, i2h))
    y_pred = np.dot(H, beta)
    y_pred = np.array([np.argmax(y_pred[i, :]) for i in range(len(y_pred))])

    print(classification_report(test_y, y_pred))
    print('Accuracy Score: {}'.format(accuracy_score(test_y, y_pred)))

    pass