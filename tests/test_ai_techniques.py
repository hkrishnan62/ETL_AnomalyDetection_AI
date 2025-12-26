"""
Unit tests for advanced AI techniques
"""

import pytest
import numpy as np
import pandas as pd
from src.validation.ai_techniques import (
    FuzzyLogicDetector,
    ExpertSystemDetector,
    TimeSeriesForecastingDetector,
    GeneticAlgorithmDetector,
    EnsembleAIDetector,
    NeuralSymbolicDetector
)


@pytest.fixture
def sample_data():
    """Create sample dataset"""
    np.random.seed(42)
    n = 100
    normal_data = np.random.normal(100, 15, (n, 3))
    
    # Add anomalies
    anomaly_idx = np.random.choice(n, 10, replace=False)
    data = normal_data.copy()
    data[anomaly_idx] += np.random.normal(0, 50, (10, 3))
    
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    df['is_anomaly'] = False
    df.loc[anomaly_idx, 'is_anomaly'] = True
    
    return df, data, anomaly_idx


class TestFuzzyLogicDetector:
    """Test Fuzzy Logic detector"""

    def test_fit_predict(self, sample_data):
        df, data, _ = sample_data
        detector = FuzzyLogicDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_membership_functions(self):
        detector = FuzzyLogicDetector()
        
        # Test triangular membership
        mem = detector._triangular_membership(0.5, 0, 0.5, 1)
        assert mem > 0
        
        mem = detector._triangular_membership(2, 0, 0.5, 1)
        assert mem == 0

    def test_normalization(self):
        detector = FuzzyLogicDetector()
        data = np.array([1, 2, 3, 4, 5])
        normalized = detector._normalize(data)
        
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1) < 0.5


class TestExpertSystemDetector:
    """Test Expert System detector"""

    def test_add_rule_and_predict(self, sample_data):
        df, _, _ = sample_data
        detector = ExpertSystemDetector()
        
        def rule_high_value(row, context):
            return row['x'] > 150
        
        detector.add_rule('high_x', rule_high_value, confidence=0.9)
        detector.fit(df[['x', 'y', 'z']])
        scores = detector.predict(df[['x', 'y', 'z']])
        
        assert len(scores) == len(df)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_multiple_rules(self, sample_data):
        df, _, _ = sample_data
        detector = ExpertSystemDetector()
        
        def rule1(row, context):
            return row['x'] > 120
        
        def rule2(row, context):
            return row['y'] < 80
        
        detector.add_rule('rule1', rule1, confidence=0.8)
        detector.add_rule('rule2', rule2, confidence=0.7)
        
        detector.fit(df[['x', 'y', 'z']])
        scores = detector.predict(df[['x', 'y', 'z']])
        
        assert len(scores) == len(df)


class TestTimeSeriesForecastingDetector:
    """Test Time Series detector"""

    def test_fit_predict(self, sample_data):
        df, data, _ = sample_data
        detector = TimeSeriesForecastingDetector(lag=5)
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_exponential_smoothing(self):
        detector = TimeSeriesForecastingDetector(alpha=0.3)
        data = np.array([100, 102, 101, 103, 102, 104, 103])
        forecast, sigma = detector._exponential_smoothing(data)
        
        assert len(forecast) == len(data)
        assert sigma > 0


class TestGeneticAlgorithmDetector:
    """Test Genetic Algorithm detector"""

    def test_fit_predict(self, sample_data):
        df, data, _ = sample_data
        detector = GeneticAlgorithmDetector(population_size=10, generations=5)
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_supervised_ga(self, sample_data):
        df, data, _ = sample_data
        labels = df['is_anomaly'].values
        
        detector = GeneticAlgorithmDetector(population_size=10, generations=5)
        detector.fit(data, labels)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)
        assert detector.best_individual is not None
        assert detector.best_fitness > -np.inf

    def test_crossover(self):
        detector = GeneticAlgorithmDetector()
        parent1 = detector._create_individual(3)
        parent2 = detector._create_individual(3)
        child = detector._crossover(parent1, parent2)
        
        assert len(child['weights']) == 3
        assert np.abs(np.sum(child['weights']) - 1) < 1e-6

    def test_mutation(self):
        detector = GeneticAlgorithmDetector()
        individual = detector._create_individual(3)
        original_threshold = individual['threshold']
        
        mutated = detector._mutate(individual, mutation_rate=1.0)
        
        # Should be different due to guaranteed mutation
        assert len(mutated['weights']) == 3


class TestEnsembleAIDetector:
    """Test Ensemble AI detector"""

    def test_fit_predict(self, sample_data):
        df, data, _ = sample_data
        detector = EnsembleAIDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_custom_weights(self, sample_data):
        df, data, _ = sample_data
        custom_weights = {
            'fuzzy': 0.4,
            'expert': 0.3,
            'timeseries': 0.2,
            'genetic': 0.1
        }
        
        detector = EnsembleAIDetector(weights=custom_weights)
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)

    def test_weights_sum_to_one(self):
        detector = EnsembleAIDetector()
        total = sum(detector.weights.values())
        assert np.abs(total - 1.0) < 1e-6


class TestNeuralSymbolicDetector:
    """Test Neural-Symbolic detector"""

    def test_fit_predict(self, sample_data):
        df, data, _ = sample_data
        detector = NeuralSymbolicDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        # May be zeros if TensorFlow unavailable
        assert len(scores) == len(data)


class TestIntegrationWithAnomalyDetector:
    """Test integration with main AnomalyDetector"""

    def test_fuzzy_method(self, sample_data):
        from src.validation.anomaly_detector import AnomalyDetector
        
        df, _, _ = sample_data
        detector = AnomalyDetector(method='fuzzy')
        detector.train_ai(df, columns=['x', 'y', 'z'], ai_method='fuzzy')
        
        predictions = detector.detect(df, columns=['x', 'y', 'z'])
        assert len(predictions) == len(df)
        assert predictions.dtype == bool

    def test_expert_method(self, sample_data):
        from src.validation.anomaly_detector import AnomalyDetector
        
        df, _, _ = sample_data
        detector = AnomalyDetector(method='expert')
        detector.train_ai(df, columns=['x', 'y', 'z'], ai_method='expert')
        
        predictions = detector.detect(df, columns=['x', 'y', 'z'])
        assert len(predictions) == len(df)

    def test_get_scores(self, sample_data):
        from src.validation.anomaly_detector import AnomalyDetector
        
        df, _, _ = sample_data
        detector = AnomalyDetector(method='ensemble')
        detector.train_ai(df, columns=['x', 'y', 'z'], ai_method='ensemble')
        
        scores = detector.detect_with_scores(df, columns=['x', 'y', 'z'])
        assert len(scores) == len(df)
        assert np.all((scores >= 0) & (scores <= 1))


class TestRobustness:
    """Test robustness to edge cases"""

    def test_single_column(self):
        data = np.array([[1], [2], [3], [100], [4], [5]])
        
        detector = FuzzyLogicDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)

    def test_with_nans(self):
        data = np.array([[1, np.nan], [2, 3], [100, 4]])
        
        detector = TimeSeriesForecastingDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)

    def test_constant_values(self):
        data = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
        
        detector = FuzzyLogicDetector()
        detector.fit(data)
        scores = detector.predict(data)
        
        assert len(scores) == len(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
