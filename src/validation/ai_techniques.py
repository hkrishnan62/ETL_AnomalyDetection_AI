"""
Advanced AI Techniques for Anomaly Detection
Beyond standard ML: Fuzzy Logic, Expert Systems, Time Series Forecasting, Genetic Algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FuzzyLogicDetector:
    """
    Fuzzy Logic-based anomaly detection using membership functions.
    Handles soft boundaries instead of hard thresholds.
    """

    def __init__(self, fuzzy_sets: Optional[Dict] = None):
        self.fuzzy_sets = fuzzy_sets or {
            'low': {'type': 'triangular', 'params': (-np.inf, -1, 0)},
            'normal': {'type': 'triangular', 'params': (-1, 0, 1)},
            'high': {'type': 'triangular', 'params': (0, 1, np.inf)}
        }
        self.scaler_params = {}

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-3, 3] range"""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / (std + 1e-8) * 3

    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a + 1e-8)
        return (c - x) / (c - b + 1e-8)

    def _gaussian_membership(self, x: float, mean: float, std: float) -> float:
        """Gaussian membership function"""
        return np.exp(-((x - mean) ** 2) / (2 * std ** 2 + 1e-8))

    def fit(self, X: np.ndarray):
        """Store normalization parameters"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for i in range(X.shape[1]):
            col_data = X[:, i]
            col_data = col_data[~np.isnan(col_data)]
            self.scaler_params[i] = {
                'mean': np.mean(col_data),
                'std': np.std(col_data)
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores [0, 1] using fuzzy membership functions.
        Higher score = more anomalous
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        anomaly_scores = np.zeros(X.shape[0])

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            normalized = self._normalize(col_data)

            for row_idx in range(len(normalized)):
                val = normalized[row_idx]
                if np.isnan(val):
                    anomaly_scores[row_idx] += 0.5
                    continue

                # Calculate membership in each fuzzy set
                memberships = {}
                for fuzzy_name, fuzzy_def in self.fuzzy_sets.items():
                    if fuzzy_def['type'] == 'triangular':
                        memberships[fuzzy_name] = self._triangular_membership(
                            val, fuzzy_def['params'][0], 
                            fuzzy_def['params'][1], 
                            fuzzy_def['params'][2]
                        )
                    elif fuzzy_def['type'] == 'gaussian':
                        memberships[fuzzy_name] = self._gaussian_membership(
                            val, fuzzy_def['params'][0], fuzzy_def['params'][1]
                        )

                # Anomaly score: inverse of normal membership
                normal_membership = memberships.get('normal', 0)
                anomaly_scores[row_idx] += (1 - normal_membership) / X.shape[1]

        return anomaly_scores


class ExpertSystemDetector:
    """
    Expert System-based detection using domain rules and inference engine.
    Combines multiple knowledge sources for contextual anomaly detection.
    """

    def __init__(self):
        self.rules = []
        self.weights = {}
        self.context_vars = {}

    def add_rule(self, name: str, condition_func, confidence: float = 1.0):
        """
        Add a domain expert rule
        condition_func: callable that takes a row and returns True/False
        confidence: how much to trust this rule [0, 1]
        """
        self.rules.append({
            'name': name,
            'condition': condition_func,
            'confidence': confidence
        })
        self.weights[name] = confidence

    def fit(self, X: pd.DataFrame):
        """Extract context from training data"""
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            col_data = X[col].dropna()
            self.context_vars[col] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'median': col_data.median(),
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75),
                'min': col_data.min(),
                'max': col_data.max()
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores combining expert rules via inference engine
        """
        scores = np.zeros(len(X))

        for idx, row in X.iterrows():
            rule_scores = []

            for rule in self.rules:
                try:
                    # Evaluate rule with row context
                    rule_context = {
                        'row': row,
                        'context': self.context_vars,
                        'index': idx
                    }
                    result = rule['condition'](row, rule_context)
                    
                    if result:
                        rule_scores.append(rule['confidence'])
                except Exception as e:
                    # Rule evaluation failed, neutral score
                    rule_scores.append(0.5)

            # Combine rules: weighted average
            if rule_scores:
                scores[idx] = np.mean(rule_scores)
            else:
                scores[idx] = 0.0

        return scores


class TimeSeriesForecastingDetector:
    """
    Time Series Forecasting based anomaly detection.
    Uses ARIMA-like approach for temporal patterns.
    """

    def __init__(self, lag: int = 5, alpha: float = 0.3):
        self.lag = lag
        self.alpha = alpha  # Exponential smoothing factor
        self.forecasts = {}
        self.residuals = {}

    def _exponential_smoothing(self, series: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple exponential smoothing forecast"""
        if len(series) < 2:
            return series, np.std(series) + 1e-8

        forecast = np.zeros(len(series))
        forecast[0] = series[0]

        for t in range(1, len(series)):
            if np.isnan(series[t]):
                forecast[t] = forecast[t-1]
            else:
                forecast[t] = self.alpha * series[t] + (1 - self.alpha) * forecast[t-1]

        residuals = series - forecast
        residuals = residuals[~np.isnan(residuals)]
        sigma = np.std(residuals) if len(residuals) > 0 else 1.0

        return forecast, sigma

    def fit(self, X: np.ndarray):
        """Fit forecasting model on training data"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            # Fill NaN values with forward fill
            for i in range(1, len(col_data)):
                if np.isnan(col_data[i]) and not np.isnan(col_data[i-1]):
                    col_data[i] = col_data[i-1]

            forecast, sigma = self._exponential_smoothing(col_data)
            self.forecasts[col_idx] = forecast
            self.residuals[col_idx] = sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores based on forecast error
        High error = anomalous
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = np.zeros(X.shape[0])

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            forecast = self.forecasts.get(col_idx, col_data)
            sigma = self.residuals.get(col_idx, 1.0)

            # Z-score of prediction error
            errors = np.abs(col_data - forecast)
            z_scores = errors / (sigma + 1e-8)
            # Normalize to [0, 1]
            z_scores = np.minimum(z_scores / 5, 1.0)
            scores += z_scores / X.shape[1]

        return scores


class GeneticAlgorithmDetector:
    """
    Genetic Algorithm-based detector for evolving anomaly detection patterns.
    Uses population-based search for optimal thresholds and feature weights.
    """

    def __init__(self, population_size: int = 20, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_fitness = -np.inf

    def _create_individual(self, n_features: int) -> Dict:
        """Create random chromosome (feature weights + threshold)"""
        return {
            'weights': np.random.dirichlet(np.ones(n_features)),
            'threshold': np.random.uniform(0.3, 0.7)
        }

    def _fitness(self, individual: Dict, X: np.ndarray, 
                 anomaly_labels: Optional[np.ndarray] = None) -> float:
        """Evaluate individual fitness (higher is better)"""
        if anomaly_labels is None:
            # Unsupervised: minimize variance while finding outliers
            X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            scores = np.dot(X_normalized, individual['weights'])
            # Fitness: detect high variance points
            outliers = np.abs(scores) > individual['threshold']
            outlier_ratio = np.mean(outliers)
            # Target 5% outlier ratio
            return -abs(outlier_ratio - 0.05)
        else:
            # Supervised: use precision/recall
            X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            scores = np.dot(X_normalized, individual['weights'])
            predictions = scores > individual['threshold']
            tp = np.sum(predictions & anomaly_labels)
            fp = np.sum(predictions & ~anomaly_labels)
            fn = np.sum(~predictions & anomaly_labels)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            return f1

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Breed two individuals"""
        child = {
            'weights': np.zeros_like(parent1['weights']),
            'threshold': 0.5 * (parent1['threshold'] + parent2['threshold'])
        }
        # Uniform crossover for weights
        for i in range(len(parent1['weights'])):
            if np.random.random() < 0.5:
                child['weights'][i] = parent1['weights'][i]
            else:
                child['weights'][i] = parent2['weights'][i]
        
        child['weights'] = child['weights'] / np.sum(child['weights'])
        return child

    def _mutate(self, individual: Dict, mutation_rate: float = 0.1) -> Dict:
        """Apply random mutations"""
        if np.random.random() < mutation_rate:
            idx = np.random.randint(len(individual['weights']))
            individual['weights'][idx] = np.random.random()
            individual['weights'] = individual['weights'] / np.sum(individual['weights'])
        
        if np.random.random() < mutation_rate:
            individual['threshold'] = np.clip(
                individual['threshold'] + np.random.normal(0, 0.05),
                0.1, 0.9
            )
        
        return individual

    def fit(self, X: np.ndarray, anomaly_labels: Optional[np.ndarray] = None):
        """Evolve population"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Initialize population
        population = [self._create_individual(X.shape[1]) 
                     for _ in range(self.population_size)]

        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [self._fitness(ind, X, anomaly_labels) for ind in population]
            
            # Track best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx].copy()

            # Selection: tournament
            survivors = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                if fitnesses[idx1] > fitnesses[idx2]:
                    survivors.append(population[idx1].copy())
                else:
                    survivors.append(population[idx2].copy())

            # Crossover and mutation
            new_population = []
            for i in range(0, len(survivors), 2):
                if i + 1 < len(survivors):
                    child1 = self._crossover(survivors[i], survivors[i+1])
                    child2 = self._crossover(survivors[i+1], survivors[i])
                else:
                    child1, child2 = survivors[i].copy(), survivors[i].copy()

                new_population.append(self._mutate(child1))
                new_population.append(self._mutate(child2))

            population = new_population[:self.population_size]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores using best evolved individual"""
        if self.best_individual is None:
            return np.zeros(len(X) if hasattr(X, '__len__') else 1)

        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        scores = np.dot(X_normalized, self.best_individual['weights'])
        # Normalize to [0, 1]
        scores = np.abs(scores)
        scores = np.minimum(scores / (np.percentile(scores, 95) + 1e-8), 1.0)
        return scores


class EnsembleAIDetector:
    """
    Ensemble AI Detector combining multiple AI techniques.
    Uses weighted voting from fuzzy logic, expert systems, time series, and GA.
    """

    def __init__(self, weights: Optional[Dict] = None):
        self.fuzzy = FuzzyLogicDetector()
        self.expert = ExpertSystemDetector()
        self.timeseries = TimeSeriesForecastingDetector()
        self.genetic = GeneticAlgorithmDetector(population_size=15, generations=8)

        self.weights = weights or {
            'fuzzy': 0.25,
            'expert': 0.25,
            'timeseries': 0.25,
            'genetic': 0.25
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Train all AI detectors"""
        self.fuzzy.fit(X)
        if hasattr(X, 'dtypes'):
            self.expert.fit(X)
        self.timeseries.fit(X)
        self.genetic.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction using weighted average of all techniques
        """
        scores = np.zeros(len(X) if hasattr(X, '__len__') else 1)

        fuzzy_scores = self.fuzzy.predict(X)
        scores += self.weights['fuzzy'] * fuzzy_scores

        if hasattr(X, 'dtypes'):
            expert_scores = self.expert.predict(X)
            scores += self.weights['expert'] * expert_scores

        ts_scores = self.timeseries.predict(X)
        scores += self.weights['timeseries'] * ts_scores

        ga_scores = self.genetic.predict(X)
        scores += self.weights['genetic'] * ga_scores

        return scores


class NeuralSymbolicDetector:
    """
    Neuro-Symbolic AI combining neural networks with symbolic reasoning.
    Bridges deep learning with interpretable rules.
    """

    def __init__(self):
        self.neural_scores = None
        self.symbolic_rules = []
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            self.TF_AVAILABLE = True
        except:
            self.TF_AVAILABLE = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Train neural component"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self.TF_AVAILABLE:
            return

        from tensorflow import keras
        from tensorflow.keras import layers

        # Shallow neural network for feature extraction
        input_layer = keras.Input(shape=(X.shape[1],))
        x = layers.Dense(8, activation='relu')(input_layer)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train with reconstruction loss proxy
        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        if y is None:
            # Unsupervised: use distance from center
            y = np.linalg.norm(X_normalized, axis=1)
            y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

        model.fit(X_normalized, y, epochs=5, batch_size=32, verbose=0)
        self.neural_model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Neural-symbolic inference"""
        if not self.TF_AVAILABLE or not hasattr(self, 'neural_model'):
            return np.zeros(len(X) if hasattr(X, '__len__') else 1)

        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        neural_scores = self.neural_model.predict(X_normalized, verbose=0).flatten()

        return neural_scores
