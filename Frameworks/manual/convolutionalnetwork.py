import numpy as np


class cnnet:
    def __init__(input_size, hidden_size, output_size=1) -> None:
        pass

    def _sigmoid(self, x):
        return 1 / (1 + np.e ** -x)

    def _der_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def forward(self, x):
        pass

    def backward(self, ):
        pass

    def fit(self, ):
        pass

    def predict(self, ):
        pass
    