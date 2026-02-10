"""Test entropy calculation functionality."""

import pytest
import torch
import numpy as np


class TestSphericalBins:
    """Test spherical binning functions."""

    def test_get_spherical_bins_shape(self, sample_representations):
        """Test spherical bins output shape."""
        from nam_entropy.h2 import get_spherical_bins

        n_bins = 10
        bins = get_spherical_bins(sample_representations, n_bins=n_bins)

        # Should return bin assignments for each sample
        assert bins.shape[0] == sample_representations.shape[0]

    def test_get_spherical_bins_deterministic(self, sample_representations):
        """Test spherical bins are deterministic."""
        from nam_entropy.h2 import get_spherical_bins

        bins1 = get_spherical_bins(sample_representations, n_bins=10)
        bins2 = get_spherical_bins(sample_representations, n_bins=10)

        assert torch.equal(bins1, bins2), "Spherical bins should be deterministic"


class TestEntropyCalculations:
    """Test entropy calculation functions."""

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        from nam_entropy.h2 import kl_divergence

        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        q = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

        kl = kl_divergence(p, q)
        assert torch.allclose(kl, torch.tensor([0.0]), atol=1e-6)

    def test_kl_divergence_different_distributions(self):
        """Test KL divergence is positive for different distributions."""
        from nam_entropy.h2 import kl_divergence

        p = torch.tensor([[0.9, 0.1]])
        q = torch.tensor([[0.5, 0.5]])

        kl = kl_divergence(p, q)
        assert kl.item() > 0, "KL divergence should be positive for different distributions"

    def test_normalize_by_scaling(self):
        """Test distribution normalization by scaling."""
        from nam_entropy.h2 import normalize_by_scaling

        unnormalized = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        normalized = normalize_by_scaling(unnormalized)

        assert torch.allclose(normalized.sum(dim=-1), torch.tensor([1.0]))

    def test_normalize_by_softmax(self):
        """Test distribution normalization by softmax."""
        from nam_entropy.h2 import normalize_by_softmax

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        normalized = normalize_by_softmax(logits)

        assert torch.allclose(normalized.sum(dim=-1), torch.tensor([1.0]))
        assert (normalized > 0).all(), "Softmax output should be positive"


class TestEntropyAccumulator2:
    """Test EntropyAccumulator2 class."""

    def test_accumulator_initialization(self):
        """Test accumulator initializes correctly."""
        from nam_entropy.h2 import EntropyAccumulator2

        acc = EntropyAccumulator2(n_bins=10)
        assert acc.n_bins == 10
        assert acc.n_samples == 0

    def test_accumulator_update(self, sample_representations_with_labels):
        """Test accumulator update method."""
        from nam_entropy.h2 import EntropyAccumulator2

        representations, labels = sample_representations_with_labels
        acc = EntropyAccumulator2(n_bins=10)

        acc.update(representations, labels)

        assert acc.n_samples == len(representations)

    def test_accumulator_multiple_updates(self, sample_representations_with_labels):
        """Test accumulator handles multiple updates."""
        from nam_entropy.h2 import EntropyAccumulator2

        representations, labels = sample_representations_with_labels
        acc = EntropyAccumulator2(n_bins=10)

        # Split data and update twice
        mid = len(representations) // 2
        acc.update(representations[:mid], labels[:mid])
        acc.update(representations[mid:], labels[mid:])

        assert acc.n_samples == len(representations)

    def test_accumulator_save_load(self, sample_representations_with_labels, tmp_path):
        """Test accumulator state can be saved and loaded."""
        from nam_entropy.h2 import EntropyAccumulator2

        representations, labels = sample_representations_with_labels
        acc = EntropyAccumulator2(n_bins=10)
        acc.update(representations, labels)

        # Save state
        save_path = tmp_path / "accumulator_state.pkl"
        acc.save_state(str(save_path))

        # Load into new accumulator
        acc2 = EntropyAccumulator2.load_state(str(save_path))

        assert acc2.n_samples == acc.n_samples
        assert acc2.n_bins == acc.n_bins

    def test_accumulator_merge(self, sample_representations_with_labels):
        """Test merging two accumulators."""
        from nam_entropy.h2 import EntropyAccumulator2

        representations, labels = sample_representations_with_labels
        mid = len(representations) // 2

        acc1 = EntropyAccumulator2(n_bins=10)
        acc1.update(representations[:mid], labels[:mid])

        acc2 = EntropyAccumulator2(n_bins=10)
        acc2.update(representations[mid:], labels[mid:])

        # Merge acc2 into acc1
        acc1.merge(acc2)

        assert acc1.n_samples == len(representations)


class TestEntropyAccumulatorValidation:
    """Test accumulator validation routines."""

    def test_validate_accumulator_sums(self, sample_representations_with_labels):
        """Test accumulator sum validation."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.validation import validate_accumulator_sums

        representations, labels = sample_representations_with_labels
        acc = EntropyAccumulator2(n_bins=10)
        acc.update(representations, labels)

        # Should not raise an error
        is_valid = validate_accumulator_sums(acc)
        assert is_valid, "Accumulator sums should be valid after update"
