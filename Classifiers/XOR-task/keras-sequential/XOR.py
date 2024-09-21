import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from matplotlib.colors import ListedColormap
from tensorflow import keras

warnings.filterwarnings('ignore')


def plot_decision_boundary(func, X, y, figsize=(9, 6)):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    c = func(ab)
    cc = c.reshape(aa.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(figsize=figsize)
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)

    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)


points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
classes = [1.0, 0.0, 0.0, 1.0]

random.seed(10)
tf.random.set_seed(10)

model = keras.Sequential([
    keras.layers.Dense(16, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=0.05, epsilon=1e-07), metrics=['AUC'])

history = model.fit(points, classes, epochs=100, verbose=0, batch_size=4)

plt.plot(history.history['loss'])
plt.show()

results = model.predict(np.array(points))
results = [0 if a < 0.5 else 1 for a in results]

plot_decision_boundary(model.predict, np.array(points), results)
plt.show()
