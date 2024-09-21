import logging
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_decision_boundary(func, X, y, figsize=(9, 6)):
    X = np.array(X)
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = []

    for obj in ab:
        c.append(func(obj))
    
    cc = np.array(c).reshape(aa.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, _ = plt.subplots(figsize=figsize)
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)

    ax_c = fig.colorbar(contour)
    ax_c.set_label('$P(y = 1)$')
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)

    plt.show()


class NeuronNetwork:
    def __init__(self, num_input=2, num_hidden=16, num_output=1):
        self.weights_01 = np.random.uniform(size=(num_input, num_hidden))
        self.weights_12 = np.random.uniform(size=(num_hidden, num_output))

        self.b01 = np.random.uniform(size=(1, num_hidden)) 
        self.b12 = np.random.uniform(size=(1, num_output))

        self.losses = []

    def update_weights(self, train_data, classes_true, l_r):
        
        # Calculate loss
        loss = 0.5 * (classes_true - self.output_final)**2
        self.losses.append(np.sum(loss))
        
        # Calculate error term
        error_term = (classes_true - self.output_final)
        
        # Calculate gradients
        grad01 = train_data.T.dot(((error_term * self._delsigmoid(self.output_final)) *  
                                    self.weights_12.T) * self._delsigmoid(self.hidden_out))
        grad12 = self.hidden_out.T.dot(error_term * self._delsigmoid(self.output_final))
        
        # Update weights and biases
        self.weights_01 += l_r * grad01  
        self.weights_12 += l_r * grad12
        self.b01 += np.sum(l_r * ((error_term * self._delsigmoid(self.output_final)) * 
                                  self.weights_12.T) * self._delsigmoid(self.hidden_out), axis=0)
        self.b12 += np.sum(l_r * error_term * self._delsigmoid(self.output_final), axis=0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _delsigmoid(self, x):
        return x * (1 - x)

    def forward(self, batch):
        self.hidden_ = np.dot(batch, self.weights_01) + self.b01
        self.hidden_out = self._sigmoid(self.hidden_)

        self.output_ = np.dot(self.hidden_out, self.weights_12) + self.b12
        self.output_final = self._sigmoid(self.output_)

        return self.output_final

    def classify(self, datapoint):
        datapoint = np.transpose(datapoint)

        class_pred = self.forward(datapoint)

        return class_pred[0][0]

    def train(self, train_data, classes_true, l_r, epochs):
        for _ in range(epochs):
            self.forward(train_data)
            self.update_weights(train_data, classes_true, l_r)


train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
classes_xor = np.array([[0], [1], [1], [0]])
classes = np.array([0, 1, 1, 0])

seed_value = 3
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

model = NeuronNetwork()
model.train(train_data, classes_xor, 1, 600)

print('-----------------------------------')
print(model.classify([0, 0]))
print(model.classify([0, 1]))
print(model.classify([1, 0]))
print(model.classify([1, 1]))

plot_decision_boundary(model.classify, train_data, classes)