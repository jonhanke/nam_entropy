
import pandas as pd
import numpy as np
import torch
import torch.distributions as dist
from typing import List, Union, Optional, Any
from numpy.typing import NDArray



## ==============================================================
## =============  Make Samples Dataframe Routine ================
## ==============================================================


def make_samples_dataframe_from_distributions(
    n_samples_list: List[int],
    distribution_list: List[Union[dist.Distribution, Any]],
    label_list: List[str],
    label_columns_name: str = 'label',
    data_component_name_list: Optional[List[str]] = None,
    randomize_samples: bool = True) -> pd.DataFrame:
    """
    Make a dataframe of data sampled from the given list 
    of (PyTorch or SciPy) distributions.  Here n_samples_list 
    tells us how many samples we're looking for in each distribution 
    and label_list tells us the names assigned to each label.  Here 
    the distributions must all have the same dimension.  
    Also label_columns_name and data_component_name_list are 
    optional parameters that tell us the names of the dataframe columns,
    with the default data names being range(n) where n is the common 
    dimension of the given distributions.
    If randomize_samples is True, then the rows are randomly permuted 
    to mix the sample types.
    """
    ## Input Validation:
    ## -----------------

    ## SANITY CHECK: To the lists all have the same length?   
    if len(n_samples_list) != len(distribution_list) or len(distribution_list) != len(label_list):
        raise ValueError("n_samples_list, dist_list, and label_list must have the same length")

    ## SANITY CHECK: Is at least one distribution provided?    
    if len(distribution_list) == 0:
        raise ValueError("At least one distribution must be provided")
    

    ## Determine the dimensionality of each distribution
    data_dimension_list = [_sample_from_distribution(dist, 1).shape[1]  for dist in distribution_list]
    unique_data_dimensions_list = list(set(data_dimension_list))

    ## SANITY CHECK: Do all distributions have the same dimensionality?   
    if len(unique_data_dimensions_list) != 1:
        raise TypeError(f"The given distributions are not valued in the same dimensional space -- data_dimension_list = {data_dimension_list}.")

    ## Determine the common data distribution dimension
    data_dim = unique_data_dimensions_list[0]

    
    ## Set up column names for data components
    if data_component_name_list is None:
        if data_dim == 1:
            data_component_name_list = ['data']
        else:
            data_component_name_list = [f'data_{i}' for i in range(data_dim)]
    else:
        if len(data_component_name_list) != data_dim:
            raise ValueError(f"data_component_name_list length ({len(data_component_name_list)}) "
                           f"must match distribution dimension ({data_dim})")
    

    ## Collect all samples and labels
    all_samples = []
    all_labels = []
    for n_samples, dist, label in zip(n_samples_list, distribution_list, label_list):
        if n_samples <= 0:
            continue
            
        ## Sample from distribution
        samples = _sample_from_distribution(dist, n_samples)
        
        ## Ensure samples have correct shape
        if samples.ndim == 1:
            if data_dim == 1:
                samples = samples.reshape(-1, 1)
            else:
                raise ValueError(f"Distribution produces 1D samples but expected {data_dim}D")
        elif samples.shape[1] != data_dim:
            raise ValueError(f"All distributions must have the same dimension. "
                           f"Expected {data_dim}, got {samples.shape[1]}")
        
        all_samples.append(samples)
        all_labels.extend([label] * n_samples)
    
    ## Combine all samples
    if not all_samples:
        raise ValueError("No samples were generated (all n_samples were <= 0)")
    
    all_data = np.vstack(all_samples)
    
    ## Create DataFrame
    df_dict = {}
    
    ## Add data columns
    if data_dim == 1:
        df_dict[data_component_name_list[0]] = all_data.flatten()
    else:
        for i, col_name in enumerate(data_component_name_list):
            df_dict[col_name] = all_data[:, i]
    
    ## Add label column
    df_dict[label_columns_name] = all_labels
    
    df = pd.DataFrame(df_dict)
    
    ## Randomize samples if requested
    if randomize_samples:
        df = df.sample(frac=1).reset_index(drop=True)
    
    ## Return the desired dataframe
    return df




def _sample_from_distribution(dist:  Union[dist.Distribution, Any], n_samples: int) -> NDArray[np.float64]:
    """
    Helper function to sample from either PyTorch or SciPy distribution.
    Always returns a 2D numpy array of shape (n_samples, n_features).
    """
    samples = None
    
    ## Check if it's a PyTorch distribution
    if hasattr(dist, 'sample'):
        ## PyTorch distribution
        if hasattr(torch, 'is_tensor') and callable(getattr(dist, 'sample')):
            sample_shape = torch.Size([n_samples])
            samples = dist.sample(sample_shape)
            samples = samples.detach().numpy()
    
    ## Check if it's a SciPy distribution (has rvs method)
    elif hasattr(dist, 'rvs'):
        ## SciPy distribution
        samples = dist.rvs(size=(n_samples,))

        ## Handle SciPy dimension inconsistencies
        if samples.ndim == 0:  # Single scalar sample
            samples = np.array([[samples]])  # Shape: (1, 1)
        elif samples.ndim == 1 and n_samples == 1:  # Single sample from multi-dim distribution
            samples = samples.reshape(1, -1)  # Shape: (1, n_dims)
        elif samples.ndim == 1:  # Multiple samples from 1D distribution
            samples = samples.reshape(-1, 1)  # Shape: (n_samples, 1)

        ## Convert the answer to a numpy array
        samples = np.asarray(samples)


    ## Check if it's a callable (custom sampling function)
    elif callable(dist):
        try:
            samples = dist(n_samples)
            samples = np.asarray(samples)
        except Exception as e:
            raise ValueError(f"Custom distribution function failed: {e}")
    
    else:
        raise ValueError(f"Distribution type not recognized. Must be PyTorch distribution, "
                        f"SciPy distribution, or callable. Got: {type(dist)}")
    
    ## Ensure we always return a 2D array
    if samples.ndim == 1:
        ## Reshape 1D array to 2D: (n_samples,) -> (n_samples, 1)
        samples = samples.reshape(-1, 1)
    elif samples.ndim == 0:
        ## Handle edge case of scalar (shouldn't happen with n_samples > 1, but just in case)
        samples = samples.reshape(1, 1)
    elif samples.ndim > 2:
        ## Flatten higher dimensions to 2D
        samples = samples.reshape(samples.shape[0], -1)
    
    return samples



# Example usage:
if __name__ == "__main__":
    # Example with SciPy distributions
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # Create some example distributions
    dist1 = stats.norm(loc=0, scale=1)  # Standard normal
    dist2 = stats.norm(loc=3, scale=1.5)  # Different normal
    dist3 = stats.uniform(loc=-2, scale=4)  # Uniform distribution
    
    # Sample from distributions
    df = make_dataframe_with_data_from_distributions(
        n_samples_list=[100, 80, 120],
        dist_list=[dist1, dist2, dist3],
        label_list=['normal_0', 'normal_3', 'uniform'],
        randomize_samples=True
    )
    
    print("Generated DataFrame:")
    print(df.head(10))
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nLabel counts:\n{df['label'].value_counts()}")
    
    # Example with PyTorch distributions (uncomment if torch is available)
    """
    import torch.distributions as torch_dist
    
    torch_dist1 = torch_dist.Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    torch_dist2 = torch_dist.Normal(torch.tensor([2.0, -1.0]), torch.tensor([1.5, 0.8]))
    
    df_torch = make_samples_dataframe_from_distributions(
        n_samples_list=[50, 50],
        distribution_list=[torch_dist1, torch_dist2],
        label_list=['class_A', 'class_B'],
        data_component_name_list=['x', 'y'],
        randomize_samples=True
    )
    
    print("\nGenerated DataFrame with PyTorch distributions:")
    print(df_torch.head())
    """




## ==============================================================
## =============  Distribution Factory Functions ================
## ==============================================================

def create_normal(loc=0.0, scale=1.0) -> dist.Normal:
    """Create Normal distribution"""
    return dist.Normal(loc=torch.tensor(float(loc)), scale=torch.tensor(max(float(scale), 0.01)))


def create_exponential(rate=1.0) -> dist.Exponential:
    """Create Exponential distribution"""
    return dist.Exponential(rate=torch.tensor(max(float(rate), 0.01)))


def create_gamma(concentration=1.0, rate=1.0) -> dist.Gamma:
    """Create Gamma distribution"""
    return dist.Gamma(concentration=torch.tensor(max(float(concentration), 0.01)), 
                      rate=torch.tensor(max(float(rate), 0.01)))


def create_beta(concentration1=1.0, concentration0=1.0) -> dist.Beta:
    """Create Beta distribution"""
    return dist.Beta(concentration1=torch.tensor(max(float(concentration1), 0.01)),
                     concentration0=torch.tensor(max(float(concentration0), 0.01)))


def create_uniform(low=0.0, high=1.0) -> dist.Uniform:
    """Create Uniform distribution"""
    low_val = min(float(low), float(high) - 0.01)
    return dist.Uniform(low=torch.tensor(low_val), high=torch.tensor(float(high)))


def create_laplace(loc:float=0.0, scale=1.0) -> dist.Laplace:
    """Create Laplace distribution"""
    return dist.Laplace(loc=torch.tensor(float(loc)), scale=torch.tensor(max(float(scale), 0.01)))




