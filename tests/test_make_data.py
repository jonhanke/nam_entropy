"""Test make_data.py module - synthetic data generation."""

import pytest
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd


class TestMakeSamplesDataframe:
    """Test make_samples_dataframe_from_distributions function."""

    def test_basic_gaussian_sampling(self):
        """Test sampling from Gaussian distributions."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        # Create two 2D Gaussian distributions
        dist1 = dist.MultivariateNormal(
            torch.zeros(2),
            torch.eye(2)
        )
        dist2 = dist.MultivariateNormal(
            torch.ones(2) * 3,
            torch.eye(2)
        )

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[50, 50],
            distribution_list=[dist1, dist2],
            label_list=['class_A', 'class_B'],
            randomize_samples=False
        )

        assert len(df) == 100
        assert 'label' in df.columns
        assert df['label'].value_counts()['class_A'] == 50
        assert df['label'].value_counts()['class_B'] == 50

    def test_1d_distribution(self):
        """Test sampling from 1D distribution."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist1 = dist.Normal(0, 1)
        dist2 = dist.Normal(5, 1)

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[30, 30],
            distribution_list=[dist1, dist2],
            label_list=['low', 'high']
        )

        assert len(df) == 60
        assert 'data' in df.columns or 'data_0' in df.columns

    def test_custom_column_names(self):
        """Test custom column naming."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist1 = dist.MultivariateNormal(torch.zeros(3), torch.eye(3))

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[20],
            distribution_list=[dist1],
            label_list=['only_class'],
            label_columns_name='category',
            data_component_name_list=['x', 'y', 'z']
        )

        assert 'category' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns

    def test_randomize_samples(self):
        """Test that randomize_samples shuffles rows."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist1 = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
        dist2 = dist.MultivariateNormal(torch.ones(2) * 10, torch.eye(2))

        # Without randomization, first 50 should all be class_A
        df_ordered = make_samples_dataframe_from_distributions(
            n_samples_list=[50, 50],
            distribution_list=[dist1, dist2],
            label_list=['class_A', 'class_B'],
            randomize_samples=False
        )

        # First 50 should all be class_A
        assert all(df_ordered['label'].iloc[:50] == 'class_A')

        # With randomization, should be mixed
        df_random = make_samples_dataframe_from_distributions(
            n_samples_list=[50, 50],
            distribution_list=[dist1, dist2],
            label_list=['class_A', 'class_B'],
            randomize_samples=True
        )

        # First 50 should NOT all be class_A (with very high probability)
        first_50_labels = df_random['label'].iloc[:50]
        assert not all(first_50_labels == 'class_A')

    def test_mismatched_list_lengths_raises(self):
        """Test error when list lengths don't match."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist1 = dist.Normal(0, 1)

        with pytest.raises(ValueError, match="same length"):
            make_samples_dataframe_from_distributions(
                n_samples_list=[50, 50],  # 2 items
                distribution_list=[dist1],  # 1 item
                label_list=['A', 'B']  # 2 items
            )

    def test_empty_distribution_list_raises(self):
        """Test error when no distributions provided."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        with pytest.raises(ValueError, match="At least one distribution"):
            make_samples_dataframe_from_distributions(
                n_samples_list=[],
                distribution_list=[],
                label_list=[]
            )

    def test_mismatched_dimensions_raises(self):
        """Test error when distributions have different dimensions."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist_2d = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
        dist_3d = dist.MultivariateNormal(torch.zeros(3), torch.eye(3))

        with pytest.raises(TypeError, match="same dimensional"):
            make_samples_dataframe_from_distributions(
                n_samples_list=[50, 50],
                distribution_list=[dist_2d, dist_3d],
                label_list=['A', 'B']
            )

    def test_zero_samples_skipped(self):
        """Test that zero-sample distributions are skipped."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        dist1 = dist.Normal(0, 1)
        dist2 = dist.Normal(5, 1)

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[50, 0],  # Second has zero samples
            distribution_list=[dist1, dist2],
            label_list=['A', 'B']
        )

        assert len(df) == 50
        assert 'B' not in df['label'].values


class TestScipyDistributions:
    """Test compatibility with scipy distributions."""

    def test_scipy_normal(self):
        """Test sampling from scipy normal distribution."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions
        from scipy import stats

        scipy_dist = stats.norm(loc=0, scale=1)

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[100],
            distribution_list=[scipy_dist],
            label_list=['scipy_normal']
        )

        assert len(df) == 100

    def test_scipy_multivariate_normal(self):
        """Test sampling from scipy multivariate normal."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions
        from scipy import stats

        scipy_mvn = stats.multivariate_normal(
            mean=[0, 0],
            cov=[[1, 0], [0, 1]]
        )

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[100],
            distribution_list=[scipy_mvn],
            label_list=['scipy_mvn']
        )

        assert len(df) == 100
        # Should have 2D data
        data_cols = [c for c in df.columns if c != 'label']
        assert len(data_cols) == 2


class TestUniformDistributions:
    """Test uniform distribution sampling."""

    def test_pytorch_uniform(self):
        """Test sampling from PyTorch uniform distribution."""
        from nam_entropy.make_data import make_samples_dataframe_from_distributions

        uniform_dist = dist.Uniform(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        df = make_samples_dataframe_from_distributions(
            n_samples_list=[100],
            distribution_list=[uniform_dist],
            label_list=['uniform']
        )

        assert len(df) == 100

        # Values should be in [0, 1]
        data_cols = [c for c in df.columns if c != 'label']
        for col in data_cols:
            assert df[col].min() >= 0
            assert df[col].max() <= 1
