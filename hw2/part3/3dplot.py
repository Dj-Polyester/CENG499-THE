import matplotlib.pyplot as plt
import pickle

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

dataset = pickle.load(open("data/part3_dataset.data", "rb"))

ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2])
plt.show()
