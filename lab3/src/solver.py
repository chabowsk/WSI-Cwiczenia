import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plots

class Node:
    def __init__(self, value=None, attribute=None, depth=0):
        self.value = value
        self.attribute = attribute
        self.children = {}
        self.depth = depth
        self.is_leaf = False

class ID3Solver:
    """
    Implementacja algorytmu ID3 zgodna z zadanym interfejsem.
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.tree = None

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {'max_depth': self.max_depth}

    def fit(self, X, y):
        """
        A method that fits the solver to the given data.
        X is the dataset without the class attribute.
        y contains the class attribute for each sample from X.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        self.tree = self.build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        A method that returns predicted class for each row of X.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        predictions = []
        # Iterujemy przez każdy wiersz do predykcji
        for _, row in X.iterrows():
            prediction = self.predict_single(row, self.tree)
            predictions.append(prediction)
            
        return np.array(predictions)

    def entropy(self, y):
        """Oblicza entropię dla zbioru etykiet y."""
        counts =  np.bincount(y) if np.issubdtype(y.dtype, np.integer) else y.value_counts()
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, X, y, attribute):
        """Oblicza zysk informacyjny (Information Gain) dla danego atrybutu."""
        total_entropy = self.entropy(y)
    
        values, counts = np.unique(X[attribute], return_counts=True)
        weighted_entropy = 0
        
        for v, count in zip(values, counts):
            subset_y = y[X[attribute] == v]
            weighted_entropy += (count / len(X)) * self.entropy(subset_y)
            
        return total_entropy - weighted_entropy

    def build_tree(self, X, y, depth):
        # Wyznaczamy najczęstszą klasę w aktualnym zbiorze (domyślna odpowiedź węzła)
        most_common_class = y.mode()[0]  
        node = Node(value=most_common_class, depth=depth)

        if len(np.unique(y)) == 1:
            node.is_leaf = True
            node.value = y.iloc[0]
            return node
        
        if X.shape[1] == 0:
            node.is_leaf = True
            return node

        if depth >= self.max_depth:
            node.is_leaf = True
            return node

        # Wybór najlepszego atrybutu (największy Information Gain)
        gains = {attr: self.information_gain(X, y, attr) for attr in X.columns}
        best_attribute = max(gains, key=gains.get)
        
        # Jeśli zysk jest zerowy, nie ma sensu dzielić dalej
        if gains[best_attribute] == 0:
            node.is_leaf = True
            return node

        node.attribute = best_attribute
        
        # Rekurencja dla każdej wartości najlepszego atrybutu
        for value in np.unique(X[best_attribute]):
            mask = X[best_attribute] == value
            X_subset = X[mask].drop(columns=[best_attribute]) # Usuwamy zużyty atrybut
            y_subset = y[mask]
            
            child_node = self.build_tree(X_subset, y_subset, depth + 1)
            node.children[value] = child_node
            
        return node

    def predict_single(self, row, node):
        """Rekurencyjne przeszukiwanie drzewa dla pojedynczej próbki."""
        if node.is_leaf:
            return node.value
        
        attribute_val = row.get(node.attribute)
        if attribute_val in node.children:
            return self.predict_single(row, node.children[attribute_val])
        else:
            #Nie znalezienie wartosci atrybuty skutkuje zwroceniem najczestszej wartosci z obecnego węzła
            return node.value


def preprocess_data(filepath):
    # Wczytanie danych z pliku
    df = pd.read_csv(filepath, sep=';')
    
    # 1. Dyskretyzacja WIEKU (z dni na lata -> kubełki)
    df['age_years'] = df['age'] / 365.25
    df['age_bin'] = pd.cut(df['age_years'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '60+'])
    
    # 2. Dyskretyzacja WAGI i WZROSTU (qcut dla równego podziału liczebności - kwantyle)
    df['weight_bin'] = pd.qcut(df['weight'], q=4, labels=['light', 'avg', 'heavy', 'very_heavy'])
    df['height_bin'] = pd.qcut(df['height'], q=4, labels=['short', 'avg', 'tall', 'very_tall'])
    
    # 3. Dyskretyzacja CIŚNIENIA SKURCZOWEGO I ROZKURCZOWEGO
    df['ap_hi_bin'] = pd.cut(df['ap_hi'], bins=[-np.inf, 120, 140, np.inf], labels=['norm', 'elevated', 'high'])
    df['ap_lo_bin'] = pd.cut(df['ap_lo'], bins=[-np.inf, 80, 90, np.inf], labels=['norm', 'elevated', 'high'])

    # Usuwamy oryginalne kolumny numeryczne oraz ID
    drop_cols = ['id', 'age', 'age_years', 'height', 'weight', 'ap_hi', 'ap_lo']
    df_processed = df.drop(columns=drop_cols)
    return df_processed

def main():
    csv_file = 'cardio_train.csv'
    
    print("--- 1. Przetwarzanie danych ---")
    try:
        data = preprocess_data(csv_file)
        print("Dane załadowane. Przykładowe wiersze po dyskretyzacji:")
        print(data.head())
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {csv_file}")
        return

    X = data.drop(columns=['cardio'])
    y = data['cardio']

    # Podział: Train (60%), Val (20%), Test (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    print(f"\nRozmiary zbiorów: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    print("\n--- 2. Trenowanie i walidacja (szukanie najlepszego max_depth) ---")

    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train_accs = []
    val_accs = []
    best_acc = 0
    best_depth = 0

    for depth in depths:
        solver = ID3Solver(max_depth=depth)
        
        solver.fit(X_train, y_train)
        
        acc_train = accuracy_score(y_train, solver.predict(X_train))
        acc_val = accuracy_score(y_val, solver.predict(X_val))        
        train_accs.append(acc_train)
        val_accs.append(acc_val)
        
        print(f"Depth: {depth} | Train: {acc_train:.4f} | Val: {acc_val:.4f}")
        
        if acc_val > best_acc:
            best_acc = acc_val
            best_depth = depth

    print(f"\nNajlepsza głębokość (wg Validation Set): {best_depth}")

    print("Generowanie wykresu Accuracy...")
    plots.plot_accuracy_evolution(depths, train_accs, val_accs)

    print("\n--- 3. Testowanie ostatecznego modelu i Macierz Pomyłek ---")
    final_solver = ID3Solver(max_depth=best_depth)
    final_solver.fit(X_train_full, y_train_full)
    
    final_predictions = final_solver.predict(X_test)
    final_test_acc = accuracy_score(y_test, final_predictions)
    
    print(f"Ostateczna dokładność na zbiorze testowym: {final_test_acc:.4f}")

    print("Generowanie Macierzy Pomyłek...")
    plots.plot_confusion_matrix_heatmap(y_test, final_predictions, title=f"Macierz pomyłek (Depth={best_depth})")

if __name__ == "__main__":
    main()