"""Test that all modules can be imported correctly."""

import pytest


class TestImports:
    """Test module imports."""

    def test_import_h(self):
        """Test h.py module imports."""
        from nam_entropy import h
        assert hasattr(h, 'EntropyAccumulator')
        assert hasattr(h, 'get_spherical_bins')
        assert hasattr(h, 'get_kmeans_bins')

    def test_import_h2(self):
        """Test h2.py module imports."""
        from nam_entropy import h2
        assert hasattr(h2, 'EntropyAccumulator2')
        assert hasattr(h2, 'get_spherical_bins')
        assert hasattr(h2, 'get_kmeans_bins')

    def test_import_model_config(self):
        """Test model_config.py module imports."""
        from nam_entropy.model_config import (
            ModelAnalyzerConfig,
            DatasetConfig,
            ModelConfig,
            TrainingConfig,
            EntropyEstimatorConfig,
        )

    def test_import_model_analyzer(self):
        """Test model_analyzer.py module imports."""
        from nam_entropy.model_analyzer import ModelAnalyzer

    def test_import_validation(self):
        """Test validation.py module imports."""
        from nam_entropy import validation
        assert hasattr(validation, 'validate_accumulator_sums')

    def test_package_version(self):
        """Test package has version."""
        import nam_entropy
        # Package should be importable
        assert nam_entropy is not None
