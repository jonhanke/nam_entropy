

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from numpy.typing import NDArray




def data_df_to_pytorch_data_tensors_and_labels(
    data_df: pd.DataFrame,
    label_list: Optional[List[str]] = None,
    label_column_name: str = 'label'
) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, int]]:
    """
    Takes a dataframe of data and returns:
        - a pytorch index tensor (of shape [n_samples])
        - a pytorch data tensor (of shape [n_samples, n_features])
        - a list of labels (length = n_labels)
        - a dict mapping labels to row indices (values are int in range(n_labels))

    Args:
        data_df: DataFrame containing data and labels
        label_list: Optional list to filter/order labels. If None, uses all unique labels sorted.
        label_column_name: Name of the label column in the DataFrame

    Returns:
        Tuple containing:
            - index_tensor: Label indices for each sample [n_samples]
            - data_tensor: Data values [n_samples, n_features]
            - label_list: List of unique labels
            - label_to_label_index_dict: Mapping from labels to indices
    """
    ## SANITY CHECK: Is label_column_name the name of a dataframe column
    if label_column_name not in data_df.columns:
        raise ValueError(f"The given label_column_name = {label_column_name} is not a column name in data_df.columns = {list(data_df.columns)}.")
    
    ## Make the label list, possibly filtering the dataframe as needed
    tmp_label_list = list(data_df[label_column_name].unique())
    if label_list == None:
        label_list = sorted(tmp_label_list)
    else:
        ## SANITY CHECK: Is label_list is a subset of the actual list of dataframe labels
        is_subset = set(label_list).issubset(set(tmp_label_list))
        if not is_subset:
            raise ValueError(f"The given label_list = {label_list} is not a subset of the list of dataframe labels {tmp_label_list}.")

        ## Filter the dataframe to accommodate the desired label list rows
        filtered_data_df = data_df[data_df[label_column_name].isin(label_list)]
        data_df = filtered_data_df
    
    ## Make the label mapping dictionary
    label_to_label_index_dict = {label: i  for i, label in enumerate(label_list)}
    #label_to_label_index_dict

    ## Make the index tensor
    index_tensor = torch.tensor([label_to_label_index_dict[x]  for x in data_df[label_column_name]])
    #index_tensor.shape
    
    ## Make the data tensor
    data_tensor = torch.tensor(data_df.drop('label', axis=1).to_numpy())    
    #data_tensor.shape
    
    ## Return the desired values
    return index_tensor, data_tensor, label_list, label_to_label_index_dict
    





def prepare_labeled_tensor_dataset(
    data_tensor_list: List[Union[torch.Tensor, np.ndarray]],
    input_tensor_labels_list: List,
    output_label_list: Optional[List] = None
) -> Tuple[torch.Tensor, torch.Tensor, List, Dict]:
    """
    Takes a list of PyTorch tensors or numpy arrays and corresponding labels and returns:
        - a pytorch index tensor (of shape [total_samples])
        - a pytorch data tensor (of shape [total_samples, tensor_features])
        - a list of labels (length = n_labels)
        - a dict mapping labels to row indices (values are int in range(n_labels))

    Args:
        data_tensor_list: List of PyTorch tensors or numpy arrays, each representing data samples
        input_tensor_labels_list: List of labels corresponding to each tensor/array in data_tensor_list
        output_label_list: Optional list to filter/order the labels. If None, uses all unique labels sorted.

    Returns:
        Tuple containing:
            - index_tensor: PyTorch tensor of label indices for each sample
            - data_tensor: PyTorch tensor containing all data samples stacked
            - output_label_list: List of unique labels used
            - label_to_label_index_dict: Dictionary mapping labels to their indices
    """
    # Validate inputs
    if len(data_tensor_list) != len(input_tensor_labels_list):
        raise ValueError(f"Length mismatch: data_tensor_list has {len(data_tensor_list)} items, input_tensor_labels_list has {len(input_tensor_labels_list)} items.")

    if len(data_tensor_list) == 0:
        raise ValueError("Input lists cannot be empty.")

    # Convert numpy arrays to PyTorch tensors and ensure 2D format
    converted_tensors = []
    for item in data_tensor_list:
        if isinstance(item, np.ndarray):
            tensor = torch.from_numpy(item).float()
        elif isinstance(item, torch.Tensor):
            tensor = item
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(item)}")

        # Ensure tensor is 2D (samples x features)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)  # Convert 1D to 2D with 1 feature
        elif tensor.dim() == 0:
            tensor = tensor.unsqueeze(0).unsqueeze(1)  # Convert scalar to 2D
        elif tensor.dim() > 2:
            # Flatten everything except the first dimension (samples)
            tensor = tensor.flatten(start_dim=1)

        converted_tensors.append(tensor)

    data_tensor_list = converted_tensors

    # Get unique labels from input
    tmp_label_list = list(set(input_tensor_labels_list))

    if output_label_list is None:
        output_label_list = sorted(tmp_label_list)
    else:
        # Sanity check: Is output_label_list a subset of the actual labels
        is_subset = set(output_label_list).issubset(set(tmp_label_list))
        if not is_subset:
            raise ValueError(f"The given output_label_list = {output_label_list} is not a subset of the input labels {tmp_label_list}.")

        # Filter data to accommodate the desired label list
        filtered_indices = [i for i, label in enumerate(input_tensor_labels_list) if label in output_label_list]
        data_tensor_list = [data_tensor_list[i] for i in filtered_indices]
        input_tensor_labels_list = [input_tensor_labels_list[i] for i in filtered_indices]

    # Create label mapping dictionary
    label_to_label_index_dict = {label: i for i, label in enumerate(output_label_list)}

    # Stack all tensors into a single data tensor
    data_tensor = torch.cat(data_tensor_list, dim=0)

    # Create index tensor by expanding each label for the number of samples in its corresponding tensor
    index_list = []
    for tensor, label in zip(data_tensor_list, input_tensor_labels_list):
        n_samples = tensor.shape[0]
        label_index = label_to_label_index_dict[label]
        index_list.extend([label_index] * n_samples)

    index_tensor = torch.tensor(index_list)

    return index_tensor, data_tensor, output_label_list, label_to_label_index_dict








def convert_tensor_list_to_dataframe(
    data_tensor_list: List[Union[torch.Tensor, NDArray]],
    input_tensor_labels_list: List[str],
    label_column_name: str = 'label',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Converts a list of PyTorch tensors/numpy arrays and labels to a pandas DataFrame
    suitable for use with data_df_to_pytorch_data_tensors_and_labels().

    Args:
        data_tensor_list: List of PyTorch tensors or numpy arrays, each representing data samples
        input_tensor_labels_list: List of labels corresponding to each tensor/array in data_tensor_list
        label_column_name: Name for the label column in the resulting DataFrame
        save_path: Optional path to save the DataFrame as CSV. If None, no file is saved.

    Returns:
        DataFrame with features as columns and the specified label column
    """

    # Validate inputs
    if len(data_tensor_list) != len(input_tensor_labels_list):
        raise ValueError(f"Length mismatch: data_tensor_list has {len(data_tensor_list)} items, input_tensor_labels_list has {len(input_tensor_labels_list)} items.")

    if len(data_tensor_list) == 0:
        raise ValueError("Input lists cannot be empty.")

    # Convert all tensors to numpy arrays and ensure 2D format
    all_data = []
    all_labels = []

    for tensor, label in zip(data_tensor_list, input_tensor_labels_list):
        # Convert to numpy if needed
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            numpy_array = tensor
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")

        # Ensure 2D format (samples x features)
        if numpy_array.ndim == 1:
            numpy_array = numpy_array.reshape(-1, 1)  # Convert 1D to 2D with 1 feature
        elif numpy_array.ndim == 0:
            numpy_array = numpy_array.reshape(1, 1)  # Convert scalar to 2D
        elif numpy_array.ndim > 2:
            # Flatten everything except the first dimension (samples)
            numpy_array = numpy_array.reshape(numpy_array.shape[0], -1)

        # Add data and corresponding labels
        all_data.append(numpy_array)
        n_samples = numpy_array.shape[0]
        all_labels.extend([label] * n_samples)

    # Stack all data
    stacked_data = np.vstack(all_data)

    # Create DataFrame
    n_features = stacked_data.shape[1]
    feature_columns = [f'feature_{i}' for i in range(n_features)]

    df_dict = {}
    for i, col_name in enumerate(feature_columns):
        df_dict[col_name] = stacked_data[:, i]
    df_dict[label_column_name] = all_labels

    df = pd.DataFrame(df_dict)

    # Save to file if path is provided
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"DataFrame saved to: {save_path}")

    return df