"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_network: marks tests that require network access")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require a GPU")


@pytest.fixture
def sample_representations():
    """Generate sample representation tensors for testing."""
    torch.manual_seed(42)
    # [batch_size, hidden_dim]
    return torch.randn(100, 64)


@pytest.fixture
def sample_representations_with_labels():
    """Generate sample representations with labels."""
    torch.manual_seed(42)
    representations = torch.randn(100, 64)
    labels = torch.randint(0, 3, (100,))  # 3 classes
    return representations, labels


@pytest.fixture
def sample_multilabel_data():
    """Generate sample data with multiple label dimensions (e.g., layers/heads)."""
    torch.manual_seed(42)
    # [batch_size, n_layers, hidden_dim]
    data = torch.randn(50, 4, 32)
    labels = torch.randint(0, 2, (50,))  # binary labels
    return data, labels


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Tiny model ID for fast integration tests
# prajjwal1/bert-tiny: 2 layers, 128 hidden, 2 attention heads, ~4.4M params
TINY_MODEL_ID = "prajjwal1/bert-tiny"


@pytest.fixture
def tiny_model_id():
    """Return the tiny model ID for testing."""
    return TINY_MODEL_ID


@pytest.fixture
def sample_sentences_with_labels():
    """Sample sentences with language labels for testing."""
    return [
        ("Hello world", "en"),
        ("Good morning", "en"),
        ("How are you?", "en"),
        ("Bonjour monde", "fr"),
        ("Bonsoir", "fr"),
        ("Comment allez-vous?", "fr"),
    ]


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'label': ['A', 'B', 'A', 'B', 'A', 'B']
    })
