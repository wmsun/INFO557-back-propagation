import math
from typing import List

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_g(x):
    return sigmoid(x) * (1 - sigmoid(x))

class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def of(cls, *layer_units: int):
        """Creates a single-layer feedforward neural network with the given
        number of units for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weights matrices, where the
        first is the weights from input units to hidden units, and the last is
        the weights from hidden units to output units.

        :param layer_weights: A list of weight matrices
        """
        self.layer_weights = layer_weights

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        activations = [input_matrix]
        self.activations = activations

        for weight in self.layer_weights:
            activations.append(sigmoid(np.dot(activations[-1], weight)))

        return activations[-1]

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        return np.where(self.predict(input_matrix) > 0.5, 1, 0)

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure for each input example.
        In the following description, × is matrix multiplication, ⊙ is
        element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, e_L, between last layer's activations, h_L,
        and the output matrix, y. Then calculate g as the element-wise product
        of the error and the sigmoid gradient of last layer's weighted sum
        (before the activation function), a_L.

        e_L = h_L - y
        g = (e_L ⊙ sigmoid'(a_L))⊤

        Then for each layer, l, starting from the last layer and working
        backwards to the first layer, accumulate the gradient for that layer,
        gradient_l, from g and the layer's activations, calculate the error that
        should be backpropagated from that layer, e_l, from g and the layer's
        weights, and calculate g as the element-wise product of e_l and the
        sigmoid gradient of that layer's weighted sum, a_l. Note that h_0 is
        defined to be the input matrix.

        gradient_l += (g × h_l)⊤
        e_l = (weights_l × g)⊤
        g = (e_l ⊙ sigmoid'(a_l))⊤

        When all input examples have applied their updates to the gradients,
        divide each gradient by the number of input examples, N.

        gradient_l /= N

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """

        i = len(self.layer_weights) - 1
        N = input_matrix.shape[0]
        h_L = self.predict(input_matrix)
        e_L = h_L - output_matrix
        a_L = np.dot(self.activations[i], self.layer_weights[i])
        g_L = np.multiply(e_L, sigmoid_g(a_L)).T

        # Previous layer l, l-1, ..., 1
        g = g_L
        gradient = []
        gradient_L = np.dot(g, self.activations[i]).T
        gradient.insert(0, gradient_L)

        while i >= 1:
            e = np.dot(self.layer_weights[i], g).T
            a = np.dot(self.activations[i - 1], self.layer_weights[i - 1])
            g = np.multiply(e, sigmoid_g(a)).T
            gradient.insert(0, np.dot(g, self.activations[i - 1]).T)
            i -= 1

        grad = [x / N for x in gradient]

        return grad

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        for i in range(iterations):
            gs = self.gradients(input_matrix, output_matrix)
            for j, w in zip(gs, self.layer_weights):
                w -= learning_rate * j

