import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import itertools
from matplotlib import pyplot as plt
import time
import multiprocessing as mp

DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTSFILE = "results.txt"


def Classifier2dModel(_in, _out, depth, breadth, actFunc):
    args = nn.ModuleList(
        [nn.Linear(_in, _out)] if depth == 1 else [nn.Linear(_in, breadth), actFunc] +
        [nn.Linear(breadth, breadth), actFunc]*(depth-2) +
        [nn.Linear(breadth, _out)]
    )
    return nn.Sequential(*args)


class Classifier2d():
    def __init__(
            self,
            trainData, trainLabels,
            testData, testLabels,
            _out=10,
            lr=.01,  numepochs=500,
            depth=1, breadth=None, actFunc=nn.ReLU(), batchSize=1, optimizer=torch.optim.Adam):

        self.lr = lr
        self.numepochs = numepochs
        self.model = Classifier2dModel(
            trainData.shape[1], _out, depth, breadth, actFunc).to(device=DEVICE)
        self.lossfun = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        trainData = TensorDataset(trainData, trainLabels)
        testData = TensorDataset(testData, testLabels)
        # pin_memory=True for optimization
        self.trainLoader = DataLoader(
            trainData, batch_size=batchSize, shuffle=True, pin_memory=True)
        self.testLoader = DataLoader(testData, batch_size=len(
            testLabels), shuffle=True, pin_memory=True)

    def getAcc(self, y, y_hat):
        # sm = F.softmax(y_hat, dim=1)
        return 100*torch.mean((torch.argmax(y_hat, axis=1) == y).float())

    def train(self):
        self.model.train()
        for X, y in self.trainLoader:
            X, y = X.to(device=DEVICE, non_blocking=True), y.to(
                device=DEVICE, non_blocking=True)
            # forward pass
            yHat = self.model(X)

            # compute loss
            trainLoss = self.lossfun(yHat, y)

            # backprop
            # set_to_none=True for optimization
            self.optimizer.zero_grad(set_to_none=True)
            trainLoss.backward()
            self.optimizer.step()

        trainAcc = self.getAcc(y, yHat)

        return trainLoss, trainAcc

    def eval(self):
        self.model.eval()
        with torch.inference_mode():
            for X, y in self.trainLoader:
                X, y = X.to(device=DEVICE, non_blocking=True), y.to(
                    device=DEVICE, non_blocking=True)

                # forward pass
                yHat = self.model(X)

                # compute loss, no backprop, instead eval
                testLoss = self.lossfun(yHat, y)
                testAcc = self.getAcc(y, yHat)

        return testLoss, testAcc

    def loop(self):
        self.trainaccs = torch.zeros(self.numepochs)
        self.testaccs = torch.zeros(self.numepochs)
        self.trainlosses = torch.zeros(self.numepochs)
        self.testlosses = torch.zeros(self.numepochs)
        for epochi in range(self.numepochs):
            # start = time.time()
            self.trainlosses[epochi], self.trainaccs[epochi] = self.train()
            self.testlosses[epochi], self.testaccs[epochi] = self.eval()
            # print(f"({epochi+1}/{self.numepochs}) time:{time.time()-start}, train:{self.trainaccs[epochi]}, test:{self.testaccs[epochi]}")

    def print(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.get_device())

# https://stackoverflow.com/a/5228294/10713877


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


# PREPROCESSING STAGE
# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(
    open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

configs = {
    # lr
    "lr": [.0001, .01, 1],
    # epochs
    "numepochs": [30],
    # depth (hidden params+2)
    "depth": [3],
    # breadth
    "breadth": [16, 128],
    # activation function
    "actFunc": [nn.ReLU(), nn.Sigmoid(), nn.Tanh()],
    # batchSize
    "batchSize": [463, 50004],
}


def test(
    x_train, y_train,
    x_test, y_test,
    times=10,
    **kwargs
):
    trainaccsRaw = torch.zeros(times, kwargs["numepochs"])
    testaccsRaw = torch.zeros(times, kwargs["numepochs"])
    trainlossesRaw = torch.zeros(times, kwargs["numepochs"])
    testlossesRaw = torch.zeros(times, kwargs["numepochs"])

    kwargsrepr = [str(v) for v in kwargs.values()]
    kwargsFileName = "_".join(kwargsrepr)
    kwargsLatex = " & ".join([f"${kwarg}$" for kwarg in kwargsrepr])

    for i in range(times):
        classifier = Classifier2d(
            x_train, y_train,
            x_test, y_test,
            **kwargs,
        )
        classifier.loop()
        trainaccsRaw[i] = classifier.trainaccs
        testaccsRaw[i] = classifier.testaccs
        trainlossesRaw[i] = classifier.trainlosses
        testlossesRaw[i] = classifier.testlosses
    classifier.trainaccs = torch.mean(trainaccsRaw, axis=0)
    classifier.testaccs = torch.mean(testaccsRaw, axis=0)
    classifier.trainlosses = torch.mean(trainlossesRaw, axis=0)
    classifier.testlosses = torch.mean(testlossesRaw, axis=0)

    testaccsMargin = 1.96 * \
        torch.std(testaccsRaw[:, -1])/torch.sqrt(torch.tensor(times))

    plt.plot(classifier.trainlosses.detach().numpy())
    plt.plot(classifier.testlosses.detach().numpy())
    plt.legend(["train loss", "test loss"])
    plt.savefig(f"{kwargsFileName}_loss.png")
    plt.clf()
    plt.plot(classifier.trainaccs.detach().numpy())
    plt.plot(classifier.testaccs.detach().numpy())
    plt.legend(["train acc", "test acc"])
    plt.savefig(f"{kwargsFileName}_acc.png")
    plt.clf()
    return kwargsLatex, classifier.testaccs[-1], testaccsMargin


STRLITERAL = "Learning rate & Number of epochs & Depth & Breadth & Activation function & Batch size & Test accuracy"


def iterConfigs(**configs):
    length = len(list(product_dict(**configs)))
    start = time.time()
    maxtestacc = 0
    maxconfig = None
    with open(RESULTSFILE, "w+") as f:
        print(f"Iteration & {STRLITERAL} & Time")
        print(f"\hline\n{STRLITERAL}\\\\ \hline", file=f)
        for i, kwargs in enumerate(product_dict(**configs)):
            print(f"({i+1}/{length})", end="\r")
            strLiteral, testacc = evalConfig(
                x_train, y_train,
                x_validation, y_validation,
                **kwargs
            )
            if testacc >= maxtestacc:
                maxconfig = kwargs
            print(f"({i+1}/{length}) & {strLiteral} & ${time.time()-start}$")
            print(f"{strLiteral}\\\\ \hline", file=f)
    return maxconfig


def evalConfig(
    x_train, y_train,
    x_test, y_test,
    **kwargs
):
    kwargsLatex, testacc, testaccsMargin = test(
        x_train, y_train,
        x_test, y_test,
        **kwargs
    )
    return f"{kwargsLatex} & ${testacc} \pm {testaccsMargin}$", testacc


configs = {
    # lr
    "lr": [.0001, .01, 1],
    # epochs
    "numepochs": [30],
    # depth (hidden params+2)
    "depth": [3],
    # breadth
    "breadth": [16, 128],
    # activation function
    "actFunc": [nn.ReLU(), nn.Sigmoid(), nn.Tanh()],
    # batchSize
    "batchSize": [463, 50004],
}
maxconfig = {
    # lr
    "lr": .01,
    # epochs
    "numepochs": 30,
    # depth (hidden params+2)
    "depth": 3,
    # breadth
    "breadth": 128,
    # activation function
    "actFunc": nn.Sigmoid(),
    # batchSize
    "batchSize": 463,
}

if __name__ == "__main__":
    # FIRST STEP
    # maxconfig = iterConfigs(**configs)
    # SECOND STEP
    new_x_train = torch.concat((x_train, x_validation))
    new_y_train = torch.concat((y_train, y_validation))

    start = time.time()
    print(f"{STRLITERAL} & Time")
    strLiteral, _ = evalConfig(
        new_x_train, new_y_train,
        x_test, y_test,
        **maxconfig,
    )
    print(f"maxconfig: {strLiteral} & {time.time()-start}")
