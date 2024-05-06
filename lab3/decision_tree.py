import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Inicjalizacja węzła drzewa
        self.feature = feature  # indeks cechy używanej do podziału
        self.threshold = threshold  # próg podziału
        self.left = left  # lewe poddrzewo
        self.right = right  # prawe poddrzewo
        self.value = value  # przechowuje klasę, jeśli węzeł jest liściem

    def is_leaf_node(self):
        # Sprawdzenie, czy węzeł jest liściem
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # Inicjalizacja drzewa decyzyjnego z parametrami
        self.min_samples_split = min_samples_split  # Minimalna liczba próbek potrzebna do podziału
        self.max_depth = max_depth  # Maksymalna głębokość drzewa
        self.n_features = n_features  # Liczba cech do losowego wyboru
        self.root = None  # Korzeń drzewa

    def fit(self, X, y):
        # Budowa drzewa z danych treningowych
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Rekurencyjne budowanie drzewa
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Warunki stopu: maksymalna głębokość, minimalna liczba próbek, jedna etykieta
        if depth >= self.max_depth or num_samples < self.min_samples_split or num_labels == 1:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Losowy wybór cech, jeśli określono
        feat_idxs = np.random.choice(num_features, self.n_features, replace=False) if self.n_features else range(num_features)
        
        # Znajdowanie najlepszego podziału
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        # Rekurencyjne tworzenie lewego i prawego poddrzewa
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        # Znajdowanie najlepszego podziału na podstawie przyrostu informacji
        best_gain = -1
        split_idx, split_thresh = None, None
        for idx in feat_idxs:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Obliczanie przyrostu informacji
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        # Podział próbek na podstawie progu
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Obliczanie entropii zestawu
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _traverse_tree(self, x, node):
        # Przechodzenie przez drzewo w celu klasyfikacji pojedynczej próbki
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        # Predykcja dla zestawu próbek
        return np.array([self._traverse_tree(x, self.root) for x in X])
