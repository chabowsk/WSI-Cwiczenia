import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(42)

# --- Klasy pomocnicze (Activation, Loss, Layer, MLP) ---

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        # Znak minus wynika z kierunku gradientu (do minimalizacji)
        return -2 * (y_true - y_pred) / y_true.size

class Layer:
    def __init__(self, input_size, output_size, activation_name='relu'):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        
        if activation_name == 'relu':
            self.activation = Activation.relu
            self.activation_deriv = Activation.relu_derivative
        elif activation_name == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_deriv = Activation.sigmoid_derivative
        else:
            self.activation = Activation.linear
            self.activation_deriv = Activation.linear_derivative
            
    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

class MLP:
    def __init__(self, layers_config, learning_rate=0.05):
        self.layers = []
        self.lr = learning_rate
        # Budowanie warstw zgodnie z konfiguracją
        for i in range(len(layers_config) - 1):
            # Ostatnia warstwa używa Sigmoidy (aby wartości były bliskie 0-1), reszta ReLU
            act = 'sigmoid' if i == len(layers_config) - 2 else 'relu'
            self.layers.append(Layer(layers_config[i], layers_config[i+1], act))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred):
        error = Loss.mse_derivative(y_true, y_pred)
        
        # Propagacja wsteczna przez warstwy
        for layer in reversed(self.layers):
            delta = error * layer.activation_deriv(layer.z)
            
            grad_w = np.dot(layer.input.T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            # Błąd do przekazania do niższej warstwy
            error = np.dot(delta, layer.weights.T)
            
            # Aktualizacja parametrów (Gradient Descent)
            layer.weights -= self.lr * grad_w
            layer.bias -= self.lr * grad_b

    def train(self, x_train, y_train, x_val, y_val, epochs=200, batch_size=32):
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Mieszanie danych (shuffling) przed każdą epoką
            indices = np.random.permutation(len(x_train))
            x_shuf, y_shuf = x_train[indices], y_train[indices]
            
            for i in range(0, len(x_train), batch_size):
                x_batch = x_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                self.backward(y_batch, self.forward(x_batch))
            
            # Obliczanie metryk po epoce
            train_out = self.forward(x_train)
            val_out = self.forward(x_val)
            
            history['train_loss'].append(Loss.mse(y_train, train_out))
            history['val_loss'].append(Loss.mse(y_val, val_out))
            
            # Dokładność (Accuracy) - porównujemy indeksy klas (argmax)
            val_preds = np.argmax(val_out, axis=1)
            val_true = np.argmax(y_val, axis=1)
            acc = np.mean(val_preds == val_true)
            history['val_acc'].append(acc)
            
            if epoch % 50 == 0:
                print(f"Epoka {epoch} - Loss: {history['train_loss'][-1]:.4f} - Val Acc: {acc:.2%}")
                
        return history

# --- Główna część skryptu ---

# 1. Wczytanie danych
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    df = pd.read_csv('wsi5-25Z_dataset.csv')

# 2. Przygotowanie One-Hot Encoding 

X = df.drop('quality', axis=1).values
y_labels = df['quality'].values

# One-hot encoding za pomocą pandas
y_one_hot = pd.get_dummies(y_labels).values
classes = np.sort(np.unique(y_labels)) # Lista oryginalnych etykiet, np. [0, 1, 2, 3, 4, 5]
num_classes = y_one_hot.shape[1]

print(f"Liczba klas: {num_classes}")
print(f"Kształt wyjścia (y): {y_one_hot.shape}")

# 3. Podział na zbiory
X_temp, X_test, y_temp, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 4. Konfiguracja sieci
# Ostatnia warstwa musi mieć rozmiar równy liczbie klas (num_classes)
mlp = MLP(layers_config=[X.shape[1], 32, 16, num_classes], learning_rate=0.05)

# 5. Trening
history = mlp.train(X_train, y_train, X_val, y_val, epochs=500, batch_size=32)

# --- Wizualizacja ---

plt.figure(figsize=(15, 6))

# Wykres błędu
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Trening')
plt.plot(history['val_loss'], label='Walidacja')
plt.title('Historia uczenia (MSE)')
plt.xlabel('Epoki')
plt.ylabel('Błąd')
plt.legend()

# Wykres Macierzy Pomyłek
plt.subplot(1, 2, 2)
# Predykcja na zbiorze testowym
test_probs = mlp.forward(X_test)

# Konwersja z One-Hot na etykiety klas (indeks neuronu o największej wartości)
y_pred_indices = np.argmax(test_probs, axis=1)
y_true_indices = np.argmax(y_test, axis=1)

# Mapowanie indeksów z powrotem na oryginalne etykiety (np. indeks 0 -> jakość 2)
y_pred_final = classes[y_pred_indices]
y_true_final = classes[y_true_indices]

cm = confusion_matrix(y_true_final, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.title('Macierz Pomyłek')
plt.xlabel('Przewidziana jakość')
plt.ylabel('Rzeczywista jakość')

plt.tight_layout()
plt.show()

# Podsumowanie skuteczności
accuracy = accuracy_score(y_true_final, y_pred_final)
print(f"Skuteczność modelu na zbiorze testowym: {accuracy:.2%}")