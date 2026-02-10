"""Test configuration classes."""

import pytest
import tempfile
import os
from pathlib import Path


class TestDatasetConfig:
    """Test DatasetConfig validation and methods."""

    def test_default_config(self):
        """Test default configuration is valid."""
        from nam_entropy.model_config import DatasetConfig
        config = DatasetConfig()
        assert config.task_repo == 'hcoxec/french_german_mix'
        assert config.get_source_type() == 'huggingface'

    def test_local_path_config(self):
        """Test local path configuration."""
        from nam_entropy.model_config import DatasetConfig
        config = DatasetConfig(task_repo=None, data_path='/tmp/data.csv')
        assert config.get_source_type() == 'local'

    def test_url_config(self):
        """Test URL configuration."""
        from nam_entropy.model_config import DatasetConfig
        config = DatasetConfig(task_repo=None, data_url='https://example.com/data.csv')
        assert config.get_source_type() == 'url'

    def test_multiple_sources_raises_error(self):
        """Test that specifying multiple sources raises an error."""
        from nam_entropy.model_config import DatasetConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DatasetConfig(
                task_repo='some/repo',
                data_path='/tmp/data.csv'
            )

    def test_no_source_raises_error(self):
        """Test that specifying no source raises an error."""
        from nam_entropy.model_config import DatasetConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DatasetConfig(task_repo=None, data_path=None, data_url=None)


class TestModelConfig:
    """Test ModelConfig."""

    def test_default_model_config(self):
        """Test default model configuration."""
        from nam_entropy.model_config import ModelConfig
        config = ModelConfig()
        assert config.model_id == 'distilbert/distilbert-base-multilingual-cased'


class TestModelAnalyzerConfig:
    """Test full ModelAnalyzerConfig."""

    def test_default_config(self):
        """Test default full configuration."""
        from nam_entropy.model_config import ModelAnalyzerConfig
        config = ModelAnalyzerConfig()
        assert config.dataset is not None
        assert config.model is not None
        assert config.training is not None

    def test_yaml_roundtrip(self):
        """Test saving and loading config from YAML."""
        from nam_entropy.model_config import ModelAnalyzerConfig

        config = ModelAnalyzerConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name

        try:
            loaded_config = ModelAnalyzerConfig.from_yaml(temp_path)
            assert loaded_config.model.model_id == config.model.model_id
            assert loaded_config.dataset.task_repo == config.dataset.task_repo
        finally:
            os.unlink(temp_path)
