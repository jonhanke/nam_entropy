"""Test that all modules can be imported correctly."""

import pytest


class TestImports:
    """Test module imports."""

    def test_import_h(self):
        """Test h.py module imports."""
        from nam_entropy import h
        assert hasattr(h, 'EntropyAccumulator')
        assert hasattr(h, 'soft_bin')

    def test_import_h2(self):
        """Test h2.py module imports."""
        from nam_entropy import h2
        assert hasattr(h2, 'EntropyAccumulator2')
        assert hasattr(h2, 'kl_divergence')
        assert hasattr(h2, 'get_spherical_bins')

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
        assert hasattr(validation, 'validate_numerical_integrity')
        assert hasattr(validation, 'validate_shape_consistency')

    def test_import_data_prep(self):
        """Test data_prep.py module imports."""
        from nam_entropy import data_prep
        assert hasattr(data_prep, 'data_df_to_pytorch_data_tensors_and_labels')
        assert hasattr(data_prep, 'prepare_labeled_tensor_dataset')

    def test_import_make_data(self):
        """Test make_data.py module imports."""
        from nam_entropy import make_data
        assert hasattr(make_data, 'make_samples_dataframe_from_distributions')
