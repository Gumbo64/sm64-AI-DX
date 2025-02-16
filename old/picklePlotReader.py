import pickle
import matplotlib.pyplot as plt
import numpy as np

while True:
    with open("picklePlot.pickle", 'rb') as file:
        figx = pickle.load(file)
    plt.show()


