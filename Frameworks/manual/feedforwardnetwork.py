import numpy as np


class ffnet:
    # TODO: classes; weights, out and errors null function; function add layer, typing
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size=1) -> None:
        self.weights = []
        # self.weights.append(np.random.randn(hidden_size_1, input_size))
        # self.weights.append(np.random.randn(hidden_size_2, hidden_size_1))
        # self.weights.append(np.random.randn(output_size, hidden_size_2))
        self.weights.append(np.array([[-1, 1], [2.5, 0.4], [1, -1.5]]))
        self.weights.append(np.array([[2.2, -1.4, 0.56], [0.34, 1.05, 3.1]]))
        self.weights.append(np.array([[0.75, -0.22]]))

        self.outs = []
        self.outs.append(np.zeros(input_size))
        self.outs.append(np.zeros(hidden_size_1))
        self.outs.append(np.zeros(hidden_size_2))
        self.outs.append(np.zeros(output_size))

        self.enters = []
        self.enters.append(np.zeros(hidden_size_1))
        self.enters.append(np.zeros(hidden_size_2))
        self.enters.append(np.zeros(output_size))

        self.errors = []
        self.errors.append(np.zeros(input_size))
        self.errors.append(np.zeros(hidden_size_1))
        self.errors.append(np.zeros(hidden_size_2))

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
        self.errors = [np.array([(y_true - y_net) * self._der_sigmoid(self.enters[-1][0])]).flatten()]

        for layer_reverse_idx, layer_enters in enumerate(list(reversed(self.enters))[1:-1]):
            new_layer_error = []

            for enter_idx, enter in enumerate(layer_enters):
                new_layer_error.append(np.sum([self.errors[-1] * self.weights[-1 - layer_reverse_idx][:, enter_idx]]) * self._der_sigmoid(enter))

            self.errors.append(np.array(new_layer_error).flatten())

        self.errors = list(reversed(self.errors))

        return self.errors


    def fit(self, ):
        pass

    def predict(self, ):
        pass
    

if __name__ == "__main__":
    nn = ffnet(2, 3, 2, 1)
    result = nn.forward([0.6, 0.7])
    print(nn.backward(0.9, result))