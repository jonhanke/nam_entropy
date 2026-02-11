"""Test EntropyAccumulator classes from h.py and h2.py."""

import pytest
import torch
import numpy as np


class TestEntropyAccumulatorInit:
    """Test EntropyAccumulator (h.py) initialization."""

    def test_basic_init(self):
        """Test basic initialization with required parameters."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])

        assert acc.n_bins == 10
        assert acc.label_list == ['A', 'B', 'C']
        assert acc.num_labels == 3
        assert acc.total_count == 0
        assert acc.bins is None  # Not pre-computed without embedding_dim

    def test_init_with_embedding_dim_unit_sphere(self):
        """Test initialization with embedding_dim and unit_sphere bins."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(
            n_bins=10,
            label_list=['A', 'B'],
            embedding_dim=64,
            bin_type='unit_sphere'
        )

        assert acc.bins is not None
        assert acc.bins.shape[0] == 10  # n_bins
        # Unit sphere bins should be normalized
        norms = torch.norm(acc.bins.view(10, -1), dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_init_with_embedding_dim_standard_normal(self):
        """Test initialization with standard_normal bins."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(
            n_bins=10,
            label_list=['A', 'B'],
            embedding_dim=64,
            bin_type='standard_normal'
        )

        assert acc.bins is not None
        assert acc.bins.shape[0] == 10

    def test_init_uniform_bins_not_precomputed(self):
        """Test that uniform bins are not pre-computed even with embedding_dim."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(
            n_bins=10,
            label_list=['A', 'B'],
            embedding_dim=64,
            bin_type='uniform'
        )

        # Uniform bins are data-dependent, so not pre-computed
        assert acc.bins is None

    def test_init_parameters_stored(self):
        """Test that all parameters are stored correctly."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(
            n_bins=20,
            label_list=['X', 'Y'],
            embedding_dim=128,
            n_heads=4,
            dist_fn='cosine',
            bin_type='unit_sphere',
            smoothing_fn='softmax',
            smoothing_temp=0.5
        )

        assert acc.n_bins == 20
        assert acc.n_heads == 4
        assert acc.dist_fn == 'cosine'
        assert acc.bin_type == 'unit_sphere'
        assert acc.smoothing_fn == 'softmax'
        assert acc.smoothing_temp == 0.5


class TestEntropyAccumulatorUpdate:
    """Test EntropyAccumulator update method."""

    def test_update_single_batch(self):
        """Test updating with a single batch of data."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        # Create test data: 100 samples, 32 dimensions
        data = torch.randn(100, 32)
        indices = torch.randint(0, 2, (100,))

        acc.update(data, indices)

        assert acc.total_count == 100
        assert acc.total_scores_sum is not None
        assert acc.label_scores_sum is not None
        assert acc.label_counts is not None
        assert acc.bins is not None  # Should be computed on first update

    def test_update_multiple_batches(self):
        """Test updating with multiple batches accumulates correctly."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        # First batch
        data1 = torch.randn(50, 32)
        indices1 = torch.zeros(50, dtype=torch.long)  # All label 'A'
        acc.update(data1, indices1)

        # Second batch
        data2 = torch.randn(30, 32)
        indices2 = torch.ones(30, dtype=torch.long)  # All label 'B'
        acc.update(data2, indices2)

        assert acc.total_count == 80
        assert acc.label_counts[0].item() == 50  # Label A count
        assert acc.label_counts[1].item() == 30  # Label B count

    def test_update_preserves_bins(self):
        """Test that bins are preserved across updates."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        data1 = torch.randn(50, 32)
        indices1 = torch.zeros(50, dtype=torch.long)
        acc.update(data1, indices1)
        bins_after_first = acc.bins.clone()

        data2 = torch.randn(50, 32)
        indices2 = torch.ones(50, dtype=torch.long)
        acc.update(data2, indices2)

        # Bins should be the same
        assert torch.allclose(acc.bins, bins_after_first)


class TestEntropyAccumulatorComputeMetrics:
    """Test EntropyAccumulator compute_metrics method."""

    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        # Create separable data for two classes
        data_a = torch.randn(100, 32) + torch.tensor([2.0] * 32)
        data_b = torch.randn(100, 32) - torch.tensor([2.0] * 32)
        data = torch.cat([data_a, data_b], dim=0)
        indices = torch.cat([torch.zeros(100), torch.ones(100)]).long()

        acc.update(data, indices)
        metrics = acc.compute_metrics()

        assert 'output_metrics' in metrics
        assert 'intermediate_data' in metrics
        assert 'entropy' in metrics['output_metrics']
        assert 'conditional_entropy' in metrics['output_metrics']
        assert 'mutual_information' in metrics['output_metrics']
        assert 'label_entropy_dict' in metrics['output_metrics']

    def test_compute_metrics_empty_raises(self):
        """Test that computing metrics on empty accumulator raises error."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        with pytest.raises(ValueError, match="empty accumulator"):
            acc.compute_metrics()

    def test_compute_metrics_structure(self):
        """Test that compute_metrics returns expected structure."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        # Create data
        torch.manual_seed(42)
        data = torch.randn(200, 32)
        indices = torch.cat([torch.zeros(100), torch.ones(100)]).long()
        acc.update(data, indices)

        metrics = acc.compute_metrics()

        # Check structure
        assert 'output_metrics' in metrics
        assert 'intermediate_data' in metrics
        om = metrics['output_metrics']
        assert 'entropy' in om
        assert 'conditional_entropy' in om
        assert 'mutual_information' in om

        # Check intermediate data
        idata = metrics['intermediate_data']
        assert 'prob_dist_for_total_population_tensor' in idata
        assert 'prob_dist_by_label_tensor' in idata
        assert 'tmp_bins' in idata

    def test_compute_metrics_label_entropies(self):
        """Test that per-label entropies are computed."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])

        data = torch.randn(300, 32)
        indices = torch.cat([torch.zeros(100), torch.ones(100), torch.full((100,), 2)]).long()
        acc.update(data, indices)

        metrics = acc.compute_metrics()
        entropy_dict = metrics['output_metrics']['label_entropy_dict']

        assert 'A' in entropy_dict
        assert 'B' in entropy_dict
        assert 'C' in entropy_dict
        assert 'total_population' in entropy_dict


class TestEntropyAccumulatorZeroSamples:
    """Test handling of labels with zero samples."""

    def test_compute_metrics_with_empty_label(self):
        """Test that compute_metrics handles labels with zero samples gracefully."""
        from nam_entropy.h import EntropyAccumulator

        # Create accumulator with 3 labels but only provide data for 2
        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])

        # Only provide data for labels A and B (index 0 and 1), not C (index 2)
        torch.manual_seed(42)
        data = torch.randn(200, 32)
        indices = torch.cat([torch.zeros(100), torch.ones(100)]).long()  # Only 0 and 1
        acc.update(data, indices)

        # This should NOT raise or return NaN
        metrics = acc.compute_metrics()

        # Check that we get valid numbers
        assert not np.isnan(metrics['output_metrics']['entropy'])
        assert not np.isnan(metrics['output_metrics']['conditional_entropy'])
        assert not np.isnan(metrics['output_metrics']['mutual_information'])

        # Check that label C is marked as None (no samples)
        entropy_dict = metrics['output_metrics']['label_entropy_dict']
        assert entropy_dict['A'] is not None
        assert entropy_dict['B'] is not None
        assert entropy_dict['C'] is None  # No samples for C

    def test_compute_metrics_uniform_weighting_with_empty_label(self):
        """Test uniform weighting only averages over labels with samples."""
        from nam_entropy.h import EntropyAccumulator

        acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])

        # Only provide data for label A
        torch.manual_seed(42)
        data = torch.randn(100, 32)
        indices = torch.zeros(100, dtype=torch.long)  # All label A
        acc.update(data, indices)

        # Uniform weighting should only consider label A (the only one with samples)
        metrics = acc.compute_metrics(conditional_entropy_label_weighting="uniform")

        assert not np.isnan(metrics['output_metrics']['conditional_entropy'])
        # With only one label having samples, conditional entropy equals that label's entropy
        assert metrics['output_metrics']['label_entropy_dict']['A'] is not None


class TestEntropyAccumulatorMerge:
    """Test EntropyAccumulator merge method."""

    def test_merge_basic(self):
        """Test basic merge of two accumulators."""
        from nam_entropy.h import EntropyAccumulator

        # Create two accumulators with same config
        acc1 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        acc2 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])

        # Update each with different data
        data1 = torch.randn(50, 32)
        indices1 = torch.zeros(50, dtype=torch.long)
        acc1.update(data1, indices1)

        # For acc2, we need to use the same bins as acc1
        acc2.bins = acc1.bins.clone()
        data2 = torch.randn(50, 32)
        indices2 = torch.ones(50, dtype=torch.long)
        acc2.update(data2, indices2)

        # Merge
        acc1.merge(acc2)

        assert acc1.total_count == 100
        assert acc1.label_counts[0].item() == 50
        assert acc1.label_counts[1].item() == 50

    def test_merge_incompatible_nbins_raises(self):
        """Test that merging with different n_bins raises error."""
        from nam_entropy.h import EntropyAccumulator

        acc1 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        acc2 = EntropyAccumulator(n_bins=20, label_list=['A', 'B'])

        # Initialize with data
        data = torch.randn(50, 32)
        indices = torch.zeros(50, dtype=torch.long)
        acc1.update(data, indices)
        acc2.update(data, indices)

        with pytest.raises(ValueError, match="n_bins mismatch"):
            acc1.merge(acc2)


class TestEntropyAccumulator2Init:
    """Test EntropyAccumulator2 (h2.py) initialization."""

    def test_default_init(self, capsys):
        """Test initialization with default config."""
        from nam_entropy.h2 import EntropyAccumulator2

        acc = EntropyAccumulator2()

        assert acc.n_bins == 1000  # Default from EntropyEstimatorConfig
        assert acc.label_list == []
        assert acc.bins is None

    def test_init_with_config(self, capsys):
        """Test initialization with custom config."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(
            n_bins=50,
            bin_type='unit_sphere',
            embedding_dim=64,
            initial_label_list=['X', 'Y', 'Z']
        )

        acc = EntropyAccumulator2(config=config)

        assert acc.n_bins == 50
        assert acc.label_list == ['X', 'Y', 'Z']
        assert acc.n_labels == 3
        assert acc.embedding_dim == 64
        # Unit sphere bins should be pre-computed
        assert acc.bins is not None

    def test_init_config_parameters(self, capsys):
        """Test that config parameters are extracted correctly."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(
            n_bins=100,
            n_heads=4,
            bin_type='uniform',
            dist_fn='cosine',
            smoothing_fn='softmax',
            smoothing_temp=0.1,
            label_name='category'
        )

        acc = EntropyAccumulator2(config=config)

        assert acc.n_bins == 100
        assert acc.n_heads == 4
        assert acc.bin_type == 'uniform'
        assert acc.dist_fn == 'cosine'
        assert acc.smoothing_fn == 'softmax'
        assert acc.smoothing_temp == 0.1
        assert acc.label_name == 'category'


class TestEntropyAccumulator2Update:
    """Test EntropyAccumulator2 update method."""

    def test_update_single_batch(self, capsys):
        """Test updating with a single batch."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(n_bins=10)
        acc = EntropyAccumulator2(config=config)

        # Create test data
        data = torch.randn(100, 32)
        indices = torch.randint(0, 2, (100,))
        labels = ['class_A', 'class_B']

        acc.update(data, indices, labels)

        assert acc.label_counts is not None
        assert 'class_A' in acc.label_list
        assert 'class_B' in acc.label_list

    def test_update_adds_new_labels(self, capsys):
        """Test that update adds new labels to the accumulator."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(
            n_bins=10,
            initial_label_list=['A']
        )
        acc = EntropyAccumulator2(config=config)

        assert acc.label_list == ['A']

        # Update with new label 'B'
        data = torch.randn(50, 32)
        indices = torch.zeros(50, dtype=torch.long)  # All point to first batch label
        acc.update(data, indices, ['B'])

        assert 'A' in acc.label_list
        assert 'B' in acc.label_list

    def test_update_with_masking(self, capsys):
        """Test that negative indices are masked out."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(n_bins=10)
        acc = EntropyAccumulator2(config=config)

        # Create data with some masked samples
        data = torch.randn(100, 32)
        # First 80 are valid (index 0 or 1), last 20 are masked (-1)
        indices = torch.cat([
            torch.randint(0, 2, (80,)),
            torch.full((20,), -1)
        ])
        labels = ['A', 'B']

        acc.update(data, indices, labels)

        # Total count should only include unmasked samples
        total = acc.label_counts.sum().item()
        assert total == 80

    def test_update_duplicate_labels_raises(self, capsys):
        """Test that duplicate labels in batch raise error."""
        from nam_entropy.h2 import EntropyAccumulator2
        from nam_entropy.model_config import EntropyEstimatorConfig

        config = EntropyEstimatorConfig(n_bins=10)
        acc = EntropyAccumulator2(config=config)

        data = torch.randn(50, 32)
        indices = torch.zeros(50, dtype=torch.long)

        with pytest.raises(RuntimeError, match="not unique"):
            acc.update(data, indices, ['A', 'A'])  # Duplicate!


class TestEntropyAccumulator2Repr:
    """Test EntropyAccumulator2 string representation."""

    def test_repr(self, capsys):
        """Test __repr__ returns expected string."""
        from nam_entropy.h2 import EntropyAccumulator2

        acc = EntropyAccumulator2()
        repr_str = repr(acc)

        assert 'EntropyAccumulator2' in repr_str
