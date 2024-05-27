import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        # Inicjalizacja neurona z daną liczbą wejść, opcjonalnym obciążeniem i wagami
        self.b = bias
        self.ws = np.array(weights) if weights is not None else np.random.rand(n_inputs)

    def _f(self, x):  # Funkcja aktywacji (leaky ReLU)
        return max(x * 0.1, x)

    def __call__(self, xs):  # Oblicza wyjście neuronu
        return self._f(np.dot(xs, self.ws) + self.b)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # Inicjalizacja warstwy składającej się z n_neurons, każdy z n_inputs wejściami
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, inputs):
        # Zwraca wyniki wywołania każdego neuronu w warstwie
        return np.array([neuron(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self):
        # Tworzenie sieci neuronowej z jedną warstwą wejściową, dwoma ukrytymi i jedną wyjściową
        self.input_layer = Layer(3, 3)  # 3 wejścia do 3 neuronów
        self.hidden_layer1 = Layer(3, 4)  # 3 wejścia do 4 neuronów
        self.hidden_layer2 = Layer(4, 4)  # 4 wejścia do 4 neuronów
        self.output_layer = Layer(4, 1)  # 4 wejścia do 1 neuronu (klasyfikacja binarna)

    def __call__(self, inputs):
        # Oblicza odpowiedź sieci na dane wejściowe
        x = self.input_layer(inputs)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x

def visualize_network(network):
    # Wizualizacja struktury sieci neuronowej
    G = nx.DiGraph()
    layers = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
    sizes = [3, 4, 4, 1]  # Liczba neuronów w każdej warstwie

    node_colors = []
    pos = {}  # Słownik pozycji

    # Definiowanie poziomych pozycji dla każdej warstwy
    layer_positions = [0, 2, 4, 6]  # Równomiernie rozmieszczone warstwy

    # Obliczanie pionowych pozycji, aby neurony były uporządkowane od góry do dołu
    for i, layer in enumerate(layers):
        neurons_in_layer = sizes[i]
        y_positions = np.linspace(neurons_in_layer/2 - 0.5, -neurons_in_layer/2 + 0.5, neurons_in_layer)
        for j in range(neurons_in_layer):
            node_id = f'{layer}_{j}'  # Indeksowanie od 0
            G.add_node(node_id, subset=i)
            pos[node_id] = (layer_positions[i], y_positions[j])  # Ustawianie pozycji horyzontalnie
            if i == 0:
                node_colors.append('red')  # Neurony wejściowe na czerwono
            elif i == len(layers) - 1:
                node_colors.append('green')  # Neuron wyjściowy na zielono
            else:
                node_colors.append('blue')  # Neurony ukryte na niebiesko

    # Dodawanie krawędzi między neuronami
    for i in range(len(sizes) - 1):
        for src in range(sizes[i]):
            for dst in range(sizes[i + 1]):
                G.add_edge(f'{layers[i]}_{src}', f'{layers[i+1]}_{dst}')

    # Rysowanie sieci
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=10, font_weight='bold', edge_color='gray', verticalalignment='center')
    plt.title('Neural Network Architecture')
    plt.show()

network = NeuralNetwork()
visualize_network(network)
