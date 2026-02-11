"""Test that KMeans n_init fix is in place."""

import pytest
import inspect


class TestKMeansFix:
    """Verify KMeans n_init=10 is explicitly set for sklearn compatibility."""

    def test_h_kmeans_has_n_init(self):
        """Test h.py cluster function with kmeans has explicit n_init=10."""
        from nam_entropy import h

        # Check the source code for n_init=10
        source = inspect.getsource(h)
        # The KMeans call should have n_init=10
        assert 'n_init=10' in source, (
            "h.py KMeans must have explicit n_init=10 for "
            "consistent behavior across sklearn versions"
        )

    def test_h2_kmeans_has_n_init(self):
        """Test h2.py has explicit n_init=10 for KMeans."""
        from nam_entropy import h2

        source = inspect.getsource(h2)
        assert 'n_init=10' in source, (
            "h2.py KMeans must have explicit n_init=10 for "
            "consistent behavior across sklearn versions"
        )
