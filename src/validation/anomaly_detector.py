# src/validation/anomaly_detector.py
import pandas as pd
import numpy as np
from .ml_anomaly import MLAnomaly
from .ai_techniques import (
    FuzzyLogicDetector,
    ExpertSystemDetector,
    TimeSeriesForecastingDetector,
    GeneticAlgorithmDetector,
    EnsembleAIDetector,
    NeuralSymbolicDetector
)


class AnomalyDetector:
    """
    Supports simple statistical IQR detection (default), ML-based methods, 
    and advanced AI techniques (Fuzzy Logic, Expert Systems, Time Series, GA, etc.)

    To preserve backward compatibility the constructor still accepts `factor`.
    """

    def __init__(self, factor=1.5, method=None, ml_params=None):
        self.factor = factor
        self.method = method
        self.ml_params = ml_params or {}
        self.ml_detector = None
        self.ai_detector = None
        
        # Advanced AI detectors
        self.fuzzy_detector = None
        self.expert_detector = None
        self.timeseries_detector = None
        self.genetic_detector = None
        self.ensemble_detector = None
        self.neural_symbolic_detector = None

    def detect_iqr(self, series):
        if series.empty:
            return pd.Series(dtype=bool)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr * self.factor)
        upper_bound = q3 + (iqr * self.factor)
        return (series < lower_bound) | (series > upper_bound)

    def train_ml(self, df, columns=None, method='isolation_forest'):
        """Train an ML-based detector on provided DataFrame and columns.

        Example: `train_ml(df, columns=['x','y'], method='autoencoder')`
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        X = df[columns]
        self.ml_detector = MLAnomaly(method=method, **self.ml_params)
        self.ml_detector.fit(X)
        self.method = method

    def train_ai(self, df, columns=None, ai_method='fuzzy'):
        """Train advanced AI-based detector.

        Methods: 'fuzzy', 'expert', 'timeseries', 'genetic', 'ensemble', 'neural_symbolic'
        Example: `train_ai(df, columns=['x','y'], ai_method='fuzzy')`
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        X = df[columns].values
        
        if ai_method == 'fuzzy':
            self.fuzzy_detector = FuzzyLogicDetector()
            self.fuzzy_detector.fit(X)
            self.ai_detector = self.fuzzy_detector
        elif ai_method == 'expert':
            self.expert_detector = ExpertSystemDetector()
            self.expert_detector.fit(df[columns])
            self.ai_detector = self.expert_detector
        elif ai_method == 'timeseries':
            self.timeseries_detector = TimeSeriesForecastingDetector()
            self.timeseries_detector.fit(X)
            self.ai_detector = self.timeseries_detector
        elif ai_method == 'genetic':
            self.genetic_detector = GeneticAlgorithmDetector()
            self.genetic_detector.fit(X)
            self.ai_detector = self.genetic_detector
        elif ai_method == 'ensemble':
            self.ensemble_detector = EnsembleAIDetector()
            self.ensemble_detector.fit(X)
            self.ai_detector = self.ensemble_detector
        elif ai_method == 'neural_symbolic':
            self.neural_symbolic_detector = NeuralSymbolicDetector()
            self.neural_symbolic_detector.fit(X)
            self.ai_detector = self.neural_symbolic_detector
        else:
            raise ValueError(f'Unknown AI method: {ai_method}')
        
        self.method = ai_method

    def detect(self, df, columns=None):
        """Detect anomalies using IQR (default), ML, or advanced AI methods.

        Returns a boolean Series indexed as `df` where True indicates anomaly.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()

        # If ML method specified and detector available, use it
        if self.method in ('isolation_forest', 'clustering', 'autoencoder'):
            if self.ml_detector is None:
                self.train_ml(df, columns=columns, method=self.method)
            X = df[columns]
            mask_arr = self.ml_detector.predict(X)
            return pd.Series(mask_arr, index=df.index)
        
        # If AI method specified, use it
        if self.method in ('fuzzy', 'expert', 'timeseries', 'genetic', 'ensemble', 'neural_symbolic'):
            if self.ai_detector is None:
                self.train_ai(df, columns=columns, ai_method=self.method)
            
            X_input = df[columns]
            if self.method == 'expert':
                # Expert system expects DataFrame
                scores = self.ai_detector.predict(X_input)
            else:
                # Others expect numpy array
                scores = self.ai_detector.predict(X_input.values)
            
            # Convert scores to binary anomaly labels (threshold at 0.5)
            threshold = 0.5 if self.method in ('fuzzy', 'ensemble', 'neural_symbolic') else 0.3
            mask = scores > threshold
            return pd.Series(mask, index=df.index)

        # Fallback to IQR
        mask = pd.Series(False, index=df.index)
        for col in columns:
            if col in df.columns:
                mask |= self.detect_iqr(df[col])
        return mask

    def detect_with_scores(self, df, columns=None):
        """Return anomaly scores instead of binary predictions.

        Useful for ranking and threshold tuning.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()

        if self.ai_detector is not None:
            X_input = df[columns]
            if self.method == 'expert':
                scores = self.ai_detector.predict(X_input)
            else:
                scores = self.ai_detector.predict(X_input.values)
            return pd.Series(scores, index=df.index)
        
        # Default: use IQR-based scoring
        scores = pd.Series(0.0, index=df.index)
        for col in columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - (iqr * self.factor)
                upper = q3 + (iqr * self.factor)
                # Score: how far outside bounds
                distances = np.maximum(0, lower - df[col], df[col] - upper)
                normalized = distances / (iqr + 1e-8)
                scores = np.maximum(scores, np.minimum(normalized, 1.0))
        
        return scores

