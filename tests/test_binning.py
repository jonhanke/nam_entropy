"""Test binning functionality."""

import pytest
import torch


class TestUnitCubeBins:
    """Test unit cube binning functions."""

    def test_unit_cube_bins_shape(self):
        """Test unit cube bins have correct shape."""
        from nam_entropy.h2 import unit_cube_bins

        start = torch.zeros(64)
        stop = torch.ones(64)
        n_bins = 10

        bins = unit_cube_bins(start, stop, n_bins)

        assert bins.shape == (n_bins, 64)

    def test_unit_cube_bins_range(self):
        """Test unit cube bins are within expected range."""
        from nam_entropy.h2 import unit_cube_bins

        start = torch.zeros(32)
        stop = torch.ones(32)
        n_bins = 5

        bins = unit_cube_bins(start, stop, n_bins)

        assert (bins >= 0).all(), "Bins should be >= start"
        assert (bins <= 1).all(), "Bins should be <= stop"


class TestBinAssignment:
    """Test bin assignment functions."""

    def test_assign_to_nearest_bin(self, sample_representations):
        """Test samples are assigned to nearest bin."""
        from nam_entropy.h2 import get_spherical_bins

        bins = get_spherical_bins(sample_representations, n_bins=10)

        # Each sample should have a bin assignment
        assert bins.shape[0] == sample_representations.shape[0]

    def test_bin_assignments_valid_range(self, sample_representations):
        """Test bin assignments are in valid range."""
        from nam_entropy.h2 import get_spherical_bins

        n_bins = 10
        bins = get_spherical_bins(sample_representations, n_bins=n_bins, just_bins=False)

        # Scores should sum to 1 (one-hot or soft assignment)
        if bins.dim() == 3:  # [N, 1, n_bins] format
            sums = bins.squeeze(1).sum(dim=-1)
        else:
            sums = bins.sum(dim=-1)

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestKMeansBins:
    """Test KMeans binning."""

    def test_kmeans_bins_shape(self, sample_representations):
        """Test KMeans bins have correct shape."""
        from nam_entropy.h2 import get_kmeans_bins

        n_bins = 8
        bins = get_kmeans_bins(sample_representations, n_bins=n_bins, just_bins=True)

        assert bins.shape == (n_bins, sample_representations.shape[1])

    def test_kmeans_assignments_shape(self, sample_representations):
        """Test KMeans assignments have correct shape."""
        from nam_entropy.h2 import get_kmeans_bins

        n_bins = 8
        assignments = get_kmeans_bins(sample_representations, n_bins=n_bins, just_bins=False)

        assert assignments.shape[0] == sample_representations.shape[0]
        assert assignments.shape[-1] == n_bins
