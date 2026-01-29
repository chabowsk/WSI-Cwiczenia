import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from solver import NaiveBayesDiscrete

def preprocess_text(text):
    #Pomocnicza funkcja czyszcząca tekst (małe litery, usuwanie znaków specjalnych)
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def load_and_process_data(filepath='spam.csv', vocab_size=1000):
    print(f"Wczytywanie i przetwarzanie danych (vocab_size={vocab_size})...")
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku spam.csv!")
        return None, None

    # Wybór kolumn i zmiana nazw
    df = df[['v1', 'v2']]
    df.columns = ['class', 'text']
    
    # 1. Budowa słownika najczęstszych słów
    vocabulary = {}
    for text in df['text']:
        tokens = preprocess_text(text).split()
        for token in tokens:
            vocabulary[token] = vocabulary.get(token, 0) + 1

    # Sortowanie i wybór top N słów
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    word_to_index = {word: i for i, (word, count) in enumerate(sorted_vocab)}

    # 2. Tworzenie macierzy dyskretnej (występowanie słowa: 0 lub 1)
    n_samples = len(df)
    X = np.zeros((n_samples, vocab_size), dtype=int)
    y = df['class'].values

    for idx, text in enumerate(df['text']):
        tokens = preprocess_text(text).split()
        for token in tokens:
            if token in word_to_index:
                X[idx, word_to_index[token]] = 1
                
    return X, y

def run_experiment():
    print("\n--- EKSPERYMENT 1: Walidacja Krzyżowa (1000 słów) ---")
    
    # 1. Przygotowanie danych
    X, y = load_and_process_data(vocab_size=1000)
    if X is None: return

    # 2. Konfiguracja Walidacji Krzyżowej (5-fold)
    k_folds = 5
    n_samples = len(y)
    indices = np.arange(n_samples)
    
    # Mieszanie danych (shuffle)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // k_folds
    accuracies = []
    
    # Listy do zbierania wyników ze wszystkich iteracji (dla macierzy pomyłek)
    all_y_true = []
    all_y_pred = []
    
    classes_found = np.unique(y)

    for i in range(k_folds):
        # Wyznaczanie indeksów start/stop dla zbioru walidacyjnego
        start = i * fold_size
        end = (i + 1) * fold_size if i < k_folds - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Inicjalizacja i trening modelu
        model = NaiveBayesDiscrete()
        model.fit(X_train, y_train)
        
        # Predykcja
        y_pred = model.predict(X_val)
        
        # Zapisywanie wyników
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        print(f"Iteracja {i+1}: Dokładność = {acc*100:.2f}%")

    # 3. Wyniki końcowe
    mean_accuracy = np.mean(accuracies)
    print(f"\nŚrednia dokładność (Accuracy): {mean_accuracy*100:.2f}%")

    # 4. Rysowanie wykresów
    
    # Wykres A: Dokładność w iteracjach
    plt.figure(figsize=(10, 5))
    iter_labels = [f'Iteracja {i+1}' for i in range(k_folds)]
    bars = plt.bar(iter_labels, accuracies, color='skyblue', edgecolor='black')
    plt.axhline(y=mean_accuracy, color='red', linestyle='--', linewidth=2, label=f'Średnia: {mean_accuracy:.3f}')
    
    plt.title('Wyniki 5-krotnej walidacji krzyżowej (1000 słów)')
    plt.ylabel('Dokładność')
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    # Wykres B: Macierz Pomyłek
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes_found)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_found)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f'Macierz pomyłek (Suma z {k_folds} iteracji)')
    plt.show()

def run_vocab_size_experiment():
    """Nowy eksperyment: Badanie wpływu liczby słów na dokładność."""
    print("\n--- EKSPERYMENT 2: Wpływ rozmiaru słownika ---")
    
    sizes = [50, 100, 200, 500, 1000]
    results = []

    for size in sizes:
        # Ponowne wczytanie i przetworzenie dla innej liczby słów
        X, y = load_and_process_data(vocab_size=size)
        if X is None: continue
        
        # Szybka 5-krotna walidacja
        k_folds = 5
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n_samples // k_folds
        accuracies = []

        for i in range(k_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k_folds - 1 else n_samples
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            model = NaiveBayesDiscrete()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracies.append(accuracy_score(y_val, y_pred))
        
        mean_acc = np.mean(accuracies)
        results.append(mean_acc)
        print(f"-> Rozmiar słownika: {size}, Średnia dokładność: {mean_acc*100:.2f}%")

    # Wykres C: Wpływ rozmiaru słownika
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, results, marker='o', linestyle='-', color='green', linewidth=2)
    plt.title('Wpływ liczby cech (rozmiaru słownika) na dokładność modelu')
    plt.xlabel('Liczba najpopularniejszych słów')
    plt.ylabel('Średnia dokładność (5-fold CV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sizes)
    
    for x, y_val in zip(sizes, results):
        plt.text(x, y_val + 0.001, f'{y_val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()          
    run_vocab_size_experiment() 