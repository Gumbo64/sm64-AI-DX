import pickle
import matplotlib.pyplot as plt

while True:
    try:
        with open('picklePlot.pickle', 'rb') as f:
            fig = pickle.load(f)
            fig.canvas.draw()
            plt.show()
    except:
        continue