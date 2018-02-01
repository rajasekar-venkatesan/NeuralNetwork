#Imports
import pandas as pd
import numpy as np
from numpy.linalg import pinv
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
    print('---BATCH ELM - MULTI-CLASS---')
    fname = 'iris_plt.csv'
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

    train_X, test_X, train_y, test_y = train_test_split(feats, labels, test_size=0.2, random_state=42)
    print('Divided into training and testing set')

    train_y_bipolar = to_bipolar(train_y, labels2index_map)
    test_y_bipolar = to_bipolar(test_y, labels2index_map)
    print('Converted labels to bipolar')

    nHidden = 10
    input_dim = train_X.shape[1]
    hidden_dim = nHidden
    output_dim = train_y_bipolar.shape[1]
    i2h = np.random.uniform(-1, 1, (input_dim, hidden_dim))

    #ELM Training
    H = sigmoid(np.dot(train_X, i2h))
    h2o = np.dot(pinv(H), train_y_bipolar)

    #ELM Testing
    H = sigmoid(np.dot(test_X, i2h))
    y_pred = np.dot(H, h2o)
    y_pred = np.array([np.argmax(y_pred[i, :]) for i in range(len(y_pred))])

    print(classification_report(test_y, y_pred))
    print('Accuracy Score: {}'.format(accuracy_score(test_y, y_pred)))

    pass