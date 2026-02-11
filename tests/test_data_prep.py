"""Test data_prep.py module - data preparation utilities."""

import pytest
import torch
import numpy as np
import pandas as pd


class TestDataDfToPytorchTensors:
    """Test data_df_to_pytorch_data_tensors_and_labels function."""

    def test_basic_conversion(self):
        """Test basic DataFrame to tensor conversion."""
        from nam_entropy.data_prep import data_df_to_pytorch_data_tensors_and_labels

        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'label': ['A', 'B', 'A', 'B']
        })

        index_tensor, data_tensor, label_list, label_dict = \
            data_df_to_pytorch_data_tensors_and_labels(df)

        assert index_tensor.shape == (4,)
        assert data_tensor.shape == (4, 2)
        assert len(label_list) == 2
        assert 'A' in label_dict and 'B' in label_dict

    def test_label_filtering(self):
        """Test filtering by label list."""
        from nam_entropy.data_prep import data_df_to_pytorch_data_tensors_and_labels

        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'label': ['A', 'B', 'C', 'A', 'B']
        })

        # Only keep A and B
        index_tensor, data_tensor, label_list, label_dict = \
            data_df_to_pytorch_data_tensors_and_labels(df, label_list=['A', 'B'])

        assert len(label_list) == 2
        assert 'C' not in label_dict
        # Should only have 4 samples (excluding the 'C' row)
        assert data_tensor.shape[0] == 4

    def test_missing_label_column_raises(self):
        """Test error when label column doesn't exist."""
        from nam_entropy.data_prep import data_df_to_pytorch_data_tensors_and_labels

        df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'wrong_column': ['A', 'B']
        })

        with pytest.raises(ValueError, match="not a column name"):
            data_df_to_pytorch_data_tensors_and_labels(df)

    def test_invalid_label_list_raises(self):
        """Test error when label_list contains invalid labels."""
        from nam_entropy.data_prep import data_df_to_pytorch_data_tensors_and_labels

        df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'label': ['A', 'B']
        })

        with pytest.raises(ValueError, match="not a subset"):
            data_df_to_pytorch_data_tensors_and_labels(df, label_list=['A', 'C'])


class TestPrepareLabeledTensorDataset:
    """Test prepare_labeled_tensor_dataset function."""

    def test_basic_tensor_preparation(self):
        """Test basic tensor list preparation."""
        from nam_entropy.data_prep import prepare_labeled_tensor_dataset

        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(15, 5)

        index_tensor, data_tensor, label_list, label_dict = \
            prepare_labeled_tensor_dataset(
                data_tensor_list=[tensor1, tensor2],
                input_tensor_labels_list=['class_A', 'class_B']
            )

        assert data_tensor.shape == (25, 5)
        assert index_tensor.shape == (25,)
        assert len(label_list) == 2

    def test_numpy_array_input(self):
        """Test with numpy arrays as input."""
        from nam_entropy.data_prep import prepare_labeled_tensor_dataset

        arr1 = np.random.randn(10, 5)
        arr2 = np.random.randn(15, 5)

        index_tensor, data_tensor, label_list, label_dict = \
            prepare_labeled_tensor_dataset(
                data_tensor_list=[arr1, arr2],
                input_tensor_labels_list=['A', 'B']
            )

        assert isinstance(data_tensor, torch.Tensor)
        assert data_tensor.shape == (25, 5)

    def test_length_mismatch_raises(self):
        """Test error when tensor and label lists have different lengths."""
        from nam_entropy.data_prep import prepare_labeled_tensor_dataset

        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(15, 5)

        with pytest.raises(ValueError, match="Length mismatch"):
            prepare_labeled_tensor_dataset(
                data_tensor_list=[tensor1, tensor2],
                input_tensor_labels_list=['A']  # Only one label!
            )

    def test_output_label_filtering(self):
        """Test filtering with output_label_list."""
        from nam_entropy.data_prep import prepare_labeled_tensor_dataset

        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(15, 5)
        tensor3 = torch.randn(20, 5)

        index_tensor, data_tensor, label_list, label_dict = \
            prepare_labeled_tensor_dataset(
                data_tensor_list=[tensor1, tensor2, tensor3],
                input_tensor_labels_list=['A', 'B', 'C'],
                output_label_list=['A', 'C']
            )

        assert len(label_list) == 2
        assert 'B' not in label_dict
        # Should only have 30 samples (A + C)
        assert data_tensor.shape[0] == 30

    def test_preserves_label_order(self):
        """Test that output preserves specified label order."""
        from nam_entropy.data_prep import prepare_labeled_tensor_dataset

        tensor1 = torch.randn(10, 5)
        tensor2 = torch.randn(15, 5)

        _, _, label_list, label_dict = \
            prepare_labeled_tensor_dataset(
                data_tensor_list=[tensor1, tensor2],
                input_tensor_labels_list=['B', 'A'],
                output_label_list=['A', 'B']  # Reversed order
            )

        # Label order should match output_label_list
        assert label_list == ['A', 'B']
        assert label_dict['A'] == 0
        assert label_dict['B'] == 1
