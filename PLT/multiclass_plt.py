#Imports
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


#Classes
class PMCELM():
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = None
        self.i2h = Variable(torch.Tensor(self.input_dim, self.hidden_dim).uniform_(-0.5, 0.5))
        self.h2o = None
        self.labels_set = None
        self.sigmoid = nn.Sigmoid()
        self.sofmax = nn.Softmax()

    def init_block(self, X, y):
        self.labels_set = list(set(y))
        self.output_dim = len(self.labels_set)
        self.h2o = Variable(torch.Tensor(self.hidden_dim, self.output_dim).uniform_(-1, 1))
        y_bipolar = self.to_bipolar(y)
        self.hidden_output = self.sigmoid(torch.mm(X, self.i2h))
        H = self.hidden_output
        self.M = pseudo_inv(torch.mm(torch.t(H), H))
        beta = torch.mm(pseudo_inv(H), y_bipolar)
        self.h2o.data = beta.data
        self.beta2a = pseudo_inv(H)

    def to_bipolar(self, label_numeric):
        label_bipolar = -1 * torch.ones(len(label_numeric), self.output_dim)
        for i in range(len(label_numeric)):
            label_bipolar[i, label_numeric[i]] = 1
        return Variable(label_bipolar)

    def forward(self, X):
        self.hidden_output = self.sigmoid(torch.mm(X, self.i2h))
        final_output_val, final_output_class = self.sofmax(torch.mm(self.hidden_output, self.h2o)).max(1)
        return final_output_class

    def update(self, X, label_numeric):
        label_bipolar = self.to_bipolar(label_numeric)
        _ = self.forward(X)
        H = self.hidden_output.data
        M = self.M.data
        M = M - torch.mm(torch.mm(torch.mm(M, torch.t(H)),
                                  torch.inverse(torch.eye(minibatch) + torch.mm(torch.mm(H, M), torch.t(H)))),
                         torch.mm(H, M))
        beta1 = self.h2o.data
        self.beta2a = torch.mm(M, torch.t(H))
        beta2 = torch.mm(self.beta2a, (label_bipolar.data - torch.mm(H, self.h2o.data)))
        self.h2o.data = beta1+beta2

    def adapt(self, new_class, minibatch):
        nclass = len(new_class)
        self.labels_set = list(set(self.labels_set).union(set(new_class)))
        print('{} new class(es) encountered'.format(nclass))
        self.output_dim = len(self.labels_set)
        delta_beta_tmp = Variable(torch.ones(minibatch, nclass)*-1)
        delta_beta = torch.mm(self.beta2a, delta_beta_tmp)
        beta = self.h2o.data
        dbeta = delta_beta.data
        new_beta = torch.cat((beta, dbeta), 1)
        self.h2o = Variable(new_beta)



#Functions
def load_data_from_csv(fname, label_column_at_end=True):
    data = pd.read_csv(fname)
    if label_column_at_end:
        label_column_name = data.columns[-1]
    else:
        label_column_name = data.columns[0]
    labels = data[label_column_name].values
    features = data.drop([label_column_name], axis=1).values
    return features, labels

def pseudo_inv(A):
    return torch.mm(torch.inverse(torch.mm(torch.t(A), A)), torch.t(A))

#Main
if __name__ == '__main__':

    fname = 'iris_plt.csv'
    feats, labels = load_data_from_csv(fname)
    num_samples = feats.shape[0]
    scaler = MinMaxScaler()
    feats = scaler.fit_transform(feats)

    model = PMCELM(4, 10)
    N0 = 100
    minibatch = 2

    init_block_X = torch.FloatTensor(feats[:N0, :]).contiguous()
    init_block_X = init_block_X.view(N0, -1)
    init_block_y = labels[:N0].tolist()

    model.init_block(Variable(init_block_X), init_block_y)

    de = N0

    while de+minibatch < num_samples-20:
        ds = de
        de = de+minibatch
        X = torch.FloatTensor(feats[ds:de, :]).contiguous().view(minibatch, -1)
        y = labels[ds:de].tolist()
        blabel = list(set(y))
        new_class = list(set(blabel) - set(model.labels_set))
        inc_class = len(new_class)
        if inc_class:
            model.adapt(new_class, minibatch)
        model.update(Variable(X), y)

    #Testing
    testX = torch.FloatTensor(feats[-20:, :]).contiguous().view(20, -1)
    y_pred = model.forward(Variable(testX)).data.numpy()
    y_act = labels[-20:]
    acc = accuracy_score(y_act, y_pred)
    ydiff = y_act - y_pred
    print('Acc: {}'.format(acc))
    pass