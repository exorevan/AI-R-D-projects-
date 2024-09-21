import numpy as np
import numpy.typing as npt


class ffnet:
    # TODO: classes; weights, out and errors null function; function add layer, typing
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        output_size: int = 1,
    ) -> None:
        self.input_size: int = input_size
        self.hidden_size_1: int = hidden_size_1
        self.hidden_size_2: int = hidden_size_2
        self.output_size: int = output_size

        self.weights: list[npt.NDArray[np.float64]] = []
        # self.weights.append(np.random.randn(hidden_size_1, input_size))
        # self.weights.append(np.random.randn(hidden_size_2, hidden_size_1))
        # self.weights.append(np.random.randn(output_size, hidden_size_2))
        self.weights.append(np.array([[-1, 1], [2.5, 0.4], [1, -1.5]]))
        self.weights.append(np.array([[2.2, -1.4, 0.56], [0.34, 1.05, 3.1]]))
        self.weights.append(np.array([[0.75, -0.22]]))

        self.outs: list[npt.NDArray[np.float64]] = []
        self.outs.append(np.zeros(input_size))
        self.outs.append(np.zeros(hidden_size_1))
        self.outs.append(np.zeros(hidden_size_2))
        self.outs.append(np.zeros(output_size))

        self.enters: list[npt.NDArray[np.float64]] = []
        self.enters.append(np.zeros(hidden_size_1))
        self.enters.append(np.zeros(hidden_size_2))
        self.enters.append(np.zeros(output_size))

        self.errors: list[npt.NDArray[np.float64]] = []
        self.errors.append(np.zeros(input_size))
        self.errors.append(np.zeros(hidden_size_1))
        self.errors.append(np.zeros(hidden_size_2))

    def _sigmoid(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 / (1 + np.e**-x)

    def _der_sigmoid(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def forward(
        self, x: list[float] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        self.outs = []
        self.outs.append(np.zeros(self.input_size))
        self.outs.append(np.zeros(self.hidden_size_1))
        self.outs.append(np.zeros(self.hidden_size_2))
        self.outs.append(np.zeros(self.output_size))

        self.enters = []
        self.enters.append(np.zeros(self.input_size))
        self.enters.append(np.zeros(self.hidden_size_1))
        self.enters.append(np.zeros(self.hidden_size_2))
        self.enters.append(np.zeros(self.output_size))

        self.outs[0] = np.array(x)
        self.enters[0] = np.array(x)

        for layer_num, weight_layer in enumerate(self.weights):
            for neuron_num, weight in enumerate(weight_layer):
                self.enters[layer_num + 1][neuron_num] = np.dot(
                    weight, self.outs[layer_num]
                )
                self.outs[layer_num + 1][neuron_num] = self._sigmoid(
                    x=self.enters[layer_num + 1][neuron_num]
                )

        return self.outs[-1]

    def backward(
        self, y_true: list[float] | npt.NDArray[np.float32]
    ) -> list[npt.NDArray[np.float64]]:
        if isinstance(y_true, list):
            y_true = np.array(y_true)

        self.errors = []
        self.errors.append(np.zeros(self.hidden_size_1))
        self.errors.append(np.zeros(self.hidden_size_2))
        self.errors.append(np.zeros(self.output_size))

        self.errors[-1] = (y_true - self.outs[-1]) * self._der_sigmoid(
            x=self.enters[-1]
        )

        for layer_num, weight_layer in enumerate(
            self.weights, start=-len(self.weights) + 1
        ):
            for neuron_num, weight in enumerate(weight_layer):
                self.errors[layer_num][neuron_num] = self._der_sigmoid(
                    self.enters[layer_num][neuron_num]
                )

    # def fit(self, X, y, epochs=100, learning_rate=0.01):
    #     for epoch in range(epochs):

    def predict(
        self, X: list[float] | npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        y_pred: npt.NDArray[np.float32] = self.forward(x=X)
        return y_pred


if __name__ == "__main__":
    nn: ffnet = ffnet(input_size=2, hidden_size_1=3, hidden_size_2=2, output_size=1)
    print(nn.forward(x=[0.6, 0.7]))
    print(nn.backward(x=[0.9]))
    # nn.fit(
    #     X=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    #     y=[0.0, 1.0, 1.0, 0.0],
    #     epochs=20,
    # )
    # result = nn.predict(X=[[0.0, 0.0]])
    # print(result)
