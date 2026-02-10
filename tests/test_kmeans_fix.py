"""Test that KMeans n_init fix is in place."""

import pytest
import inspect


class TestKMeansFix:
    """Verify KMeans n_init=10 is explicitly set for sklearn compatibility."""

    def test_h_kmeans_has_n_init(self):
        """Test h.py get_kmeans_bins has explicit n_init=10."""
        from nam_entropy import h
        source = inspect.getsource(h.get_kmeans_bins)
        assert 'n_init=10' in source, (
            "h.py get_kmeans_bins must have explicit n_init=10 for "
            "consistent behavior across sklearn versions"
        )

    def test_h2_kmeans_has_n_init(self):
        """Test h2.py get_kmeans_bins has explicit n_init=10."""
        from nam_entropy import h2
        source = inspect.getsource(h2.get_kmeans_bins)
        assert 'n_init=10' in source, (
            "h2.py get_kmeans_bins must have explicit n_init=10 for "
            "consistent behavior across sklearn versions"
        )

    def test_kmeans_produces_consistent_results(self, sample_representations):
        """Test that KMeans binning produces consistent results."""
        from nam_entropy.h2 import get_kmeans_bins
        import torch

        torch.manual_seed(42)
        bins1 = get_kmeans_bins(sample_representations, n_bins=5, just_bins=True)

        torch.manual_seed(42)
        bins2 = get_kmeans_bins(sample_representations, n_bins=5, just_bins=True)

        # Results should be identical with same seed
        assert torch.allclose(bins1, bins2), "KMeans should produce consistent results"
