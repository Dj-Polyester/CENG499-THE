from Kmeansexperiment import *


if __name__ == "__main__":
    test(dataset1, filename="results_dataset1_plus",
         initializer=KMeans.KMeansPlusPlus)
    test(dataset2, filename="results_dataset2_plus",
         initializer=KMeans.KMeansPlusPlus)
