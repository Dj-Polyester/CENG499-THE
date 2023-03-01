from Kmeansexperiment import *

filename1 = "results_dataset1"
filename2 = "results_dataset2"


def saveFigsAndResults(dataset, filename):
    filename_plus = f"{filename}_plus"
    filename_merged = f"{filename}_merged"

    plt.xlabel(K)
    plt.ylabel(LOSS)
    Ks, losses = test(dataset, filename=filename)
    plt.plot(Ks, losses.mean(axis=1))
    plt.savefig(f"{filename}.png")
    plt.clf()

    plt.xlabel(K)
    plt.ylabel(LOSS)
    plt.plot(Ks, losses.mean(axis=1))
    Ks, losses = test(dataset, filename=filename_plus,
                      initializer=KMeans.KMeansPlusPlus)
    plt.plot(Ks, losses.mean(axis=1))
    plt.legend(["KMeans", "KMeans++"])
    plt.savefig(f"{filename_merged}.png")
    plt.clf()

    plt.xlabel(K)
    plt.ylabel(LOSS)
    plt.plot(Ks, losses.mean(axis=1))
    plt.savefig(f"{filename_plus}.png")
    plt.clf()


if __name__ == "__main__":
    saveFigsAndResults(dataset1, filename1)
    saveFigsAndResults(dataset2, filename2)
