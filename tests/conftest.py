"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np


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
