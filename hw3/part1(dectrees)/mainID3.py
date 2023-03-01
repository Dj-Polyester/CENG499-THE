import numpy as np
from ID3 import DecisionTree


attributes = np.array(["Temperature", "Outlook", "Humidity", "Windy"])
# Golf played?...
labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1])
dataset = np.array([
    ["hot", "sunny", "high", "false"],  # 0
    ["hot", "sunny", "high", "true"],  # 0
    ["hot", "overcast", "high", "false"],  # 1
    ["cool", "rain", "normal", "false"],  # 1
    ["cool", "overcast", "normal", "true"],  # 1
    ["mild", "sunny", "high", "false"],  # 0
    ["cool", "sunny", "normal", "false"],  # 1
    ["mild", "rain", "normal", "false"],  # 1
    ["mild", "sunny", "normal", "true"],  # 1
    ["mild", "overcast", "high", "true"],  # 1
    ["hot", "overcast", "normal", "false"],  # 1
    ["mild", "rain", "high", "true"],  # 0
    ["cool", "rain", "normal", "true"],  # 0
    ["mild", "rain", "high", "false"]])  # 1

if __name__ == "__main__":
    dt = DecisionTree(dataset, labels, attributes)
    dt.train()
    correct = 0
    wrong = 0
    for data_index in range(len(dataset)):
        data_point = dataset[data_index]
        data_label = labels[data_index]

        predicted = dt.predict(data_point)
        if predicted == data_label:
            correct += 1
        else:
            wrong += 1

    print("Accuracy : %.2f" % (correct/(correct+wrong)*100))
    # You need graphviz to use the method below (sudo apt install graphviz)
    dt.print2dotfile("tree_infogain")
    dt.print2dotfile("tree_gainratio")
