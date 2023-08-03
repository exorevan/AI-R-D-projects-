import numpy as np


class ffnet:
    # TODO: classes; weights, out and errors null function; function add layer
    def __init__(self, input_size, hidden_size, output_size=1) -> None:
        self.weights = []
        self.weights.append(np.random.randn(hidden_size, input_size))
        self.weights.append(np.random.randn(output_size, hidden_size))

        self.outs = []
        self.outs.append(np.zeros(input_size))
        self.outs.append(np.zeros(hidden_size))
        self.outs.append(np.zeros(output_size))

        self.enters = []
        self.enters.append(np.zeros(hidden_size))
        self.enters.append(np.zeros(output_size))

        self.errors = []
        self.errors.append(np.zeros(input_size))
        self.errors.append(np.zeros(hidden_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.e ** -x)

    def _der_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def forward(self, x):
        self.outs = []
        self.enters = []
        cur_value = np.array(x)
        self.enters.append(cur_value)

        for layer in self.weights:
            self.outs.append(cur_value)

            cur_value = np.matmul(layer, cur_value)
            self.enters.append(cur_value)

            cur_value = self._sigmoid(cur_value)

        self.outs.append(cur_value)

        return cur_value

    def backward(self, y_true, y_net):
        self.errors = [[(y_true - y_net) * self._der_sigmoid(self.enters[-1][0])]]

        for layer_reverse_idx, layer_enters in enumerate(list(reversed(self.enters))[:-1]):
            for enter_idx, enter in enumerate(layer_enters):
                ...


    def fit(self, ):
        pass

    def predict(self, ):
        pass
    

if __name__ == "__main__":
    nn = ffnet(2, 3, 1)
    print(nn.forward([0.6, 0.8]))