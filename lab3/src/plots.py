import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_accuracy_evolution(depths, train_accs, val_accs):
    """
    Rysuje wykres zmian dokładności (Accuracy) w zależności od głębokości drzewa
    dla zbiorów: treningowego i walidacyjnego.
    """
    plt.figure(figsize=(10, 6))
    
    # Rysowanie linii
    plt.plot(depths, train_accs, label='Zbiór Treningowy', marker='o', linestyle='--')
    plt.plot(depths, val_accs, label='Zbiór Walidacyjny', marker='o')
    
    # Opisy
    plt.title('Zależność Accuracy od głębokości drzewa (Max Depth)')
    plt.xlabel('Maksymalna głębokość drzewa')
    plt.ylabel('Dokładność (Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wyświetlenie
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, title="Macierz Pomyłek (Confusion Matrix)"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(title)
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Rzeczywista klasa')
    
    plt.tight_layout()
    plt.show()