import numpy as np

# Funkcja sigmoid z opcją obliczania pochodnej
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights is not None:
            self.ws = np.array(weights)
        else:
            # Dodajemy jeden do liczby wejść, aby uwzględnić obciążenie
            self.ws = np.random.rand(n_inputs + 1)
        self.inputs = None
        self.output = None
        self.delta = None

    # Funkcja aktywacji (sigmoid)
    def _f(self, x):
        return sigmoid(x)

    # Obliczanie wyjścia neuronu
    def __call__(self, xs):
        # Dodajemy 1 do wejść, aby uwzględnić obciążenie
        self.inputs = np.hstack([xs, 1])
        self.output = self._f(np.dot(self.inputs, self.ws))
        return self.output

    # Obliczanie delty dla propagacji wstecznej
    def calculate_delta(self, target=None, forward_deltas=None, forward_weights=None):
        if target is not None:
            self.delta = (self.output - target) * sigmoid(self.output, derivative=True)
        else:
            self.delta = sum(forward_deltas * forward_weights) * sigmoid(self.output, derivative=True)

    # Aktualizacja wag i obciążenia
    def update_weights(self, lr):
        self.ws -= lr * self.delta * self.inputs

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    # Obliczanie wyjść dla warstwy neuronów
    def __call__(self, inputs):
        return np.array([neuron(inputs) for neuron in self.neurons])

    # Obliczanie delt dla warstwy
    def calculate_deltas(self, targets=None, next_layer=None):
        for i, neuron in enumerate(self.neurons):
            if targets is not None:
                neuron.calculate_delta(target=targets[i])
            else:
                forward_deltas = np.array([next_neuron.delta for next_neuron in next_layer.neurons])
                forward_weights = np.array([next_neuron.ws[i] for next_neuron in next_layer.neurons])
                neuron.calculate_delta(forward_deltas=forward_deltas, forward_weights=forward_weights)

    # Aktualizacja wag dla warstwy
    def update_weights(self, lr):
        for neuron in self.neurons:
            neuron.update_weights(lr)

class NeuralNetwork:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.input_layer = Layer(3, 3)  # 3 wejścia do 3 neuronów
        self.hidden_layer1 = Layer(3, 4)  # 3 wejścia do 4 neuronów
        self.hidden_layer2 = Layer(4, 4)  # 4 wejścia do 4 neuronów
        self.output_layer = Layer(4, 1)  # 4 wejścia do 1 neuronu (klasyfikacja binarna)

    # Obliczanie wyjścia sieci neuronowej
    def __call__(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x

    # Obliczanie funkcji straty
    def calculate_loss(self, predicted, target):
        return 0.5 * (target - predicted) ** 2

    # Propagacja wsteczna
    def backpropagation(self, inputs, target, lr):
        # Przepływ w przód
        predicted = self.__call__(inputs)

        # Obliczanie delt
        self.output_layer.calculate_deltas(targets=[target])
        self.hidden_layer2.calculate_deltas(next_layer=self.output_layer)
        self.hidden_layer1.calculate_deltas(next_layer=self.hidden_layer2)
        self.input_layer.calculate_deltas(next_layer=self.hidden_layer1)

        # Aktualizacja wag
        self.output_layer.update_weights(lr)
        self.hidden_layer2.update_weights(lr)
        self.hidden_layer1.update_weights(lr)
        self.input_layer.update_weights(lr)

    # Trenowanie sieci
    def train(self, X, y, num_iterations):
        for i in range(num_iterations):
            for xi, yi in zip(X, y):
                self.backpropagation(xi, yi, self.alpha)

    # Przewidywanie wyjść dla danych wejściowych
    def predict(self, X):
        return np.array([self.__call__(xi) for xi in X])

# Dane wejściowe
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])

# Oczekiwane wyjścia
y = np.array([[0], [1], [0], [1], [1], [0]])

# Inicjalizacja sieci
network = NeuralNetwork(alpha=0.1)

# Trenowanie sieci
network.train(X, y, num_iterations=10000)

# Przewidywanie i wyświetlanie wyników po treningu
print("Output After Training: \n{}".format(network.predict(X)))
