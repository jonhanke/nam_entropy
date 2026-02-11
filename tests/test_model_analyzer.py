"""Test ModelAnalyzer configuration classes."""

import pytest
import tempfile
import os


class TestModelAnalyzerImport:
    """Test ModelAnalyzer can be imported."""

    def test_import_model_analyzer(self):
        """Test ModelAnalyzer class can be imported."""
        from nam_entropy.model_analyzer import ModelAnalyzer
        assert ModelAnalyzer is not None


class TestModelAnalyzerConfig:
    """Test ModelAnalyzer configuration handling."""

    def test_default_config_creation(self):
        """Test ModelAnalyzerConfig can be created with defaults."""
        from nam_entropy.model_config import ModelAnalyzerConfig
        config = ModelAnalyzerConfig()

        assert config.model is not None
        assert config.dataset is not None
        assert config.training is not None

    def test_config_from_yaml(self, tmp_path):
        """Test config can be loaded from YAML file."""
        from nam_entropy.model_config import ModelAnalyzerConfig

        yaml_content = """
dataset:
  task_repo: "hcoxec/french_german_mix"
  data_split_name: "train"
model:
  model_id: "distilbert/distilbert-base-multilingual-cased"
training:
  batch_size: 16
  device: "cpu"
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = ModelAnalyzerConfig.from_yaml(str(yaml_path))

        assert config.dataset.task_repo == "hcoxec/french_german_mix"
        assert config.model.model_id == "distilbert/distilbert-base-multilingual-cased"
        assert config.training.batch_size == 16


class TestModelAnalyzerDatasetLoading:
    """Test dataset loading configuration."""

    def test_get_source_type_huggingface(self):
        """Test source type detection for HuggingFace."""
        from nam_entropy.model_config import DatasetConfig

        config = DatasetConfig(task_repo="some/repo")
        assert config.get_source_type() == "huggingface"

    def test_get_source_type_local(self):
        """Test source type detection for local files."""
        from nam_entropy.model_config import DatasetConfig

        config = DatasetConfig(task_repo=None, data_path="/path/to/file.csv")
        assert config.get_source_type() == "local"

    def test_get_source_type_url(self):
        """Test source type detection for URLs."""
        from nam_entropy.model_config import DatasetConfig

        config = DatasetConfig(task_repo=None, data_url="https://example.com/data.csv")
        assert config.get_source_type() == "url"


class TestTrainingConfig:
    """Test TrainingConfig."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from nam_entropy.model_config import TrainingConfig

        config = TrainingConfig()

        assert config.batch_size > 0
        assert config.device in ['cpu', 'cuda', 'mps']


class TestEntropyEstimatorConfig:
    """Test EntropyEstimatorConfig."""

    def test_entropy_estimator_config_defaults(self):
        """Test EntropyEstimatorConfig has sensible defaults."""
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig()

        assert config.n_bins > 0
        assert config.bin_type in ['unit_sphere', 'uniform']
