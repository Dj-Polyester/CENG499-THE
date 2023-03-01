from KMeans import KMeans
import pickle
import numpy as np
from matplotlib import pyplot as plt
import time
K = "K"
LOSS = "Loss"
STRLITERAL = f"{K} & {LOSS}"

dataset1 = pickle.load(open("data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("data/part2_dataset_2.data", "rb"))


def test(dataset, Ks=np.arange(2, 11), times=10, mintimes=10, initializer=KMeans.KMeansMinusMinus, filename=None):
    if 0 in Ks:
        raise Exception("K value cannot be 0")
    lenKs = len(Ks)
    losses = np.zeros((lenKs, times))
    start = time.time()
    with open(f"{filename}.txt", "w+") as f:
        print(f"Iteration & {STRLITERAL} & Time")
        print("\\hline", file=f)
        print(f"{STRLITERAL} \\\\ \\hline", file=f)
        for i in range(lenKs):
            for j in range(times):
                minVal = np.inf
                for _ in range(mintimes):
                    kmeans = KMeans(dataset, Ks[i], initializer)
                    _, __, tmp = kmeans.run()
                    minVal = min(minVal, tmp)
                losses[i, j] = minVal
            lossMargin = 1.96 * (losses[i].std()/np.sqrt(times))
            loss = losses[i].mean()
            resultTxt = f"{Ks[i]} & ${loss} \pm {lossMargin}$"

            print(f"({i+1}/{lenKs}) & {resultTxt} & ${time.time()-start}$")
            print(f"{resultTxt} \\\\ \\hline", file=f)
    return Ks, losses
    # plt.xlabel(K)
    # plt.ylabel(LOSS)
    # plt.plot(Ks, losses.mean(axis=1))
    # plt.savefig(f"{filename}.png")
    # plt.clf()


if __name__ == "__main__":
    test(dataset1, filename="results_dataset1")
    test(dataset2, filename="results_dataset2")
