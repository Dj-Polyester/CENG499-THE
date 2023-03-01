import torch
import numpy as np
'''
a simple ANN implementation
'''


def onehotencode(s, indexingFunc):
    amax = indexingFunc(s, axis=1)
    tmp = torch.zeros(s.shape)
    tmp[torch.arange(len(tmp)), amax] = 1
    return tmp


def acc(y, y_hat):
    return torch.mean(torch.sum(onehotencode(y_hat, torch.argmax)*y, axis=1))


def sigmoid(s):
    return 1/(1+torch.e**(-s))


def identity(s):
    return s


def softmax(s):
    denom = torch.sum(torch.e**s, axis=1)
    return ((torch.e**s).T/denom).T

# loss functions


def SE(y, y_hat):
    return torch.mean((y-y_hat)**2)


def CE(y, y_hat):
    return torch.mean(-torch.sum(y*torch.log(y_hat), axis=1))

# negative of derivative of mean


def nderCE(y, y_hat):
    tmp = y.clone()
    for elemy, elemy_hat in zip(tmp, y_hat):
        elemy = elemy @ (torch.eye(len(elemy)) -
                         torch.tile(elemy_hat, (len(elemy), 1)))
    return tmp/len(y)


def nderSE(y, y_hat):
    return 2*(y-y_hat)/len(y)


def nder(a):
    if a == SE:
        return nderSE
    elif a == CE:
        return nderCE
    raise Exception("negative derivative could not be found")

# derivative


def dersigmoid(O):
    return O*(1-O)


def der(a):
    if a == sigmoid:
        return dersigmoid
    raise Exception("derivative could not be found")


class Sequential():
    def __init__(
        self,
        data,
        labels,
        lossFunc,
        model: list[tuple[torch.tensor]],
        stepSize,
        batchSize,
        numepochs,
    ) -> None:
        self.data = data
        self.labels = labels
        self.lossFunc = lossFunc
        self.model = model
        self.batchSize = batchSize
        self.stepSize = stepSize
        self.numEpochs = numepochs
        if len(self.data) != len(self.labels):
            raise Exception(
                f"data and labels have different lengths {len(self.data)} and {len(self.labels)}")
        self.Os = []
        self.deltas = []

        def forward(self):
            self.indices = np.random.choice(
                torch.arange(len(self.labels)), self.batchSize, replace=False)
            self.y = self.labels[self.indices]
            O = self.data[self.indices]
            for w, b, a in self.model:
                self.Os.append(O)
                O = a(O @ w+b)
            self.O = O
            #Os is backwards

        def backward(self):
            # self.y must be in one-hot encoding for classification
            delta = nder(self.lossFunc)(self.y, self.O)
            self.deltas.append(delta)
            ws, bs, as_ = zip(*self.model)
            for w, a, O in zip(reversed(ws[1:]), reversed(as_[:-1]), reversed(self.Os[1:])):
                delta = der(a)(O)*(delta @ w.T)
                self.deltas.append(delta)
            for w, b, O, delta in zip(ws, bs, self.Os, reversed(self.deltas)):
                w += self.stepSize*(torch.transpose(O, (len(O), 1)) @ delta)
                b += self.stepSize*(torch.ones(self.batchSize) @ delta)

        def train(self):
            for _ in range(self.numEpochs):
                self.forward()
                self.backward()
