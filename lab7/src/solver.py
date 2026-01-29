import numpy as np

class NaiveBayesDiscrete:
    def __init__(self):
        self.priors = {}      # P(C)
        self.likelihoods = {} # P(xi | C)
        self.classes = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            
            self.priors[c] = X_c.shape[0] / n_samples
            
            self.likelihoods[c] = {}
            for i in range(n_features):
                self.likelihoods[c][i] = {}
                
                feature_values = np.unique(X[:, i])
                
                vals_in_class, counts = np.unique(X_c[:, i], return_counts=True)
                counts_dict = dict(zip(vals_in_class, counts))
                
                for val in feature_values:
                    count = counts_dict.get(val, 0)
                    # P(xi | C) z wygładzaniem Laplace'a (+1 w liczniku, +LiczbaWartości w mianowniku)
                    prob = (count + 1) / (X_c.shape[0] + len(feature_values))
                    self.likelihoods[c][i][val] = prob

    def predict(self, X):
        predictions = []
        for row in X:
            posteriors = {}
            for c in self.classes:
                posterior = np.log(self.priors[c])
                
                for i, val in enumerate(row):
                    if val in self.likelihoods[c][i]:
                        posterior += np.log(self.likelihoods[c][i][val])
                    else:
                        posterior += np.log(1e-6)
                
                posteriors[c] = posterior
            
            best_class = max(posteriors, key=posteriors.get)
            predictions.append(best_class)
            
        return np.array(predictions)