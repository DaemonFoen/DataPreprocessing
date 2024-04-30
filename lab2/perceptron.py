import pickle

import numpy as np


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.where(x <= 0, 0, 1)


class Lay:
    def __init__(self, input_size, lay_size):
        self.neurons = [Neuron(input_size) for _ in range(lay_size)]

    def compute(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons]).T

    def local_gradient(self, loss):
        for i, neuron in enumerate(self.neurons):
            neuron.calculate_grad(loss[i])

    def compute_weight(self, learning_rate, previous_lay_output):
        for neuron in self.neurons:
            neuron.weights += learning_rate * neuron.grad * previous_lay_output
            neuron.bias += learning_rate * neuron.grad

    def get_output(self):
        return np.array([n.output for n in self.neurons])

    def back_loss(self):
        return np.dot(np.array([neuron.grad for neuron in self.neurons]),
                      np.array([neuron.weights for neuron in self.neurons]))


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * np.sqrt(2. / input_size)
        self.bias = 0
        self.output = None
        self.input = None
        self.grad = None

    def activate(self, inputs):
        self.input = inputs
        self.output = relu(np.dot(inputs, self.weights) + self.bias)
        return self.output

    def calculate_grad(self, error):
        self.grad = error * derivative_relu(self.output)


def save_weights(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_weights(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


class Perceptron:

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Lay(input_size, hidden_size)
        self.hidden_layer2 = Lay(hidden_size, hidden_size)
        self.output_layer = Lay(hidden_size, output_size)

    def backpropagation(self, inputs, expected_output, learning_rate):
        inputs = np.array(inputs, dtype=np.float64)
        outputs = self.predict(inputs)
        loss = expected_output - outputs

        self.output_layer.local_gradient(loss)
        back_loss = self.output_layer.back_loss()

        self.hidden_layer2.local_gradient(back_loss)
        self.output_layer.compute_weight(learning_rate, self.hidden_layer2.get_output())

        self.hidden_layer2.compute_weight(learning_rate, self.hidden_layer.get_output())
        back_loss_2 = self.hidden_layer2.back_loss()
        self.hidden_layer.local_gradient(back_loss_2)
        self.hidden_layer.compute_weight(learning_rate, inputs)

    def predict(self, inputs):
        hidden_outputs = self.hidden_layer.compute(inputs)
        hidden_outputs2 = self.hidden_layer2.compute(hidden_outputs)
        return self.output_layer.compute(hidden_outputs2)

    def train(self, training_inputs, training_outputs, learning_rate, epochs):
        for _ in range(epochs):
            for inputs, expected_output in zip(training_inputs, training_outputs):
                self.backpropagation(inputs, expected_output, learning_rate)
