"""Test binning functionality."""

import pytest
import torch


class TestUnitCubeBins:
    """Test unit cube binning functions."""

    def test_unit_cube_bins_in_range(self):
        """Test unit cube bins are within expected range."""
        from nam_entropy.h2 import unit_cube_bins

        start = torch.zeros(32)
        stop = torch.ones(32)
        n_bins = 5

        bins = unit_cube_bins(start, stop, n_bins)

        assert (bins >= 0).all(), "Bins should be >= start"
        assert (bins <= 1).all(), "Bins should be <= stop"

    def test_unit_cube_bins_deterministic(self):
        """Test unit cube bins are deterministic (no randomness)."""
        from nam_entropy.h2 import unit_cube_bins

        start = torch.zeros(16)
        stop = torch.ones(16)

        bins1 = unit_cube_bins(start, stop, 5)
        bins2 = unit_cube_bins(start, stop, 5)

        assert torch.equal(bins1, bins2)


class TestSphericalBinsGeneration:
    """Test spherical bin generation."""

    def test_spherical_bins_shape(self):
        """Test spherical bins have correct shape."""
        from nam_entropy.h2 import get_spherical_bins

        bins = get_spherical_bins(
            n_bins=10,
            ambient_space_dimension=64,
            device=torch.device('cpu'),
            dtype=torch.float32
        )

        assert bins.shape == (10, 64)

    def test_spherical_bins_unit_norm(self):
        """Test spherical bins are unit normalized."""
        from nam_entropy.h2 import get_spherical_bins

        bins = get_spherical_bins(10, 32, torch.device('cpu'), torch.float32)

        norms = torch.norm(bins, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_spherical_bins_different_dims(self):
        """Test spherical bins work with different dimensions."""
        from nam_entropy.h2 import get_spherical_bins

        for dim in [16, 64, 128, 768]:
            bins = get_spherical_bins(5, dim, torch.device('cpu'), torch.float32)
            assert bins.shape == (5, dim)
