"""
Soft Entropy Calculation Module (h2.py)
=======================================

This module implements soft entropy calculation methods for analyzing neural network representations.
It provides functions for:
- Creating bins for discretizing continuous representations
- Computing soft assignments of representations to bins
- Various distance functions and smoothing techniques
- Conditional counting and entropy analysis

Key Components:
- Binning strategies (uniform, sphere, clustering)
- Distance functions (euclidean, cosine, dot product)
- Smoothing functions (softmax, sparsemax, discrete)
- Information-theoretic measures (entropy, mutual information, disentanglement)

"""

import torch
import torch.nn.functional as F

from entmax import sparsemax
from torch.distributions import Uniform
from sklearn.cluster import KMeans

from typing import Optional, Literal, Tuple

from .model_config import EntropyEstimatorConfig



## ==========================================================================
## =================== SOFT-BINNING CALCULATION ROUTINES ====================
## ==========================================================================



@torch.no_grad()
def soft_bin2(all_representations: torch.Tensor,
             n_bins: int,
             bins: Optional[torch.Tensor] = None,
             centers: Optional[torch.Tensor] = None,
             dist_fn: Literal['cosine', 'euclidean', 'dot', 'cosine_5', 'cluster'] = 'euclidean',
             bin_type: Literal['uniform', 'standard_normal', 'unit_sphere', 'unit_cube_by_bins', 'unit_cube_by_interpolation', 'cluster'] = 'uniform',
             sub_mean: bool = False,
#             n_heads: int = 1,
             extra_internal_label_dims_list = [],    ## e.g. [n_layers+1, n_heads]
             smoothing_fn: Literal["softmax", "sparsemax", "discrete", "None"] = "None",
             smoothing_temp: float = 1.0,
             online_bins: Optional[torch.Tensor] = None,
             set_var: float = 1.0,
             online_var: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             show_diagnostics: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor]:    
    """
    Performs soft binning of neural representations using various distance metrics.

    This is the main function for converting continuous neural representations into
    discrete probability distributions over bins. It supports multiple binning strategies,
    distance functions, and smoothing techniques.

    Args:
        all_representations (torch.Tensor): Input representations to bin [N, D]
        n_bins (int): Number of bins to use for discretization
        bins (torch.Tensor, optional): Pre-computed bin locations
        centers (torch.Tensor, optional): Pre-computed bin centers
        temp (float): Temperature for smoothing (default: 1.0)
        dist_fn (str): Distance function ('cosine', 'euclidean', 'dot')
        bin_type (str): Binning strategy ('uniform', 'unit_sphere', 'cluster', etc.)
        sub_mean (bool): Whether to subtract mean (default: False)
        n_heads (int): Number of attention heads for multi-head processing
        smoothing_fn (str): Smoothing function ('softmax', 'sparsemax', 'discrete')
        online_bins (torch.Tensor, optional): Cached bins from previous iterations
        set_var (float): Variance scaling factor (default: 1.0)
        online_var (tuple, optional): Cached variance statistics

    Returns:
        tuple: (scores, bins)
##            - scores: Soft assignment probabilities [N, n_heads, n_bins]
##            - scores: Soft assignment probabilities [N, n_bins]
##            - scores: Soft assignment probabilities [N, ...structural_label_dims..., n_bins]
            - scores: Soft assignment probabilities [N, *extra_internal_label_dims_list, n_bins]
            - bins: Bin locations used for scoring

    Example:
        >>> representations = torch.randn(100, 64)
        >>> scores, bins = soft_bin(representations, n_bins=50)
        >>> print(scores.shape)  # torch.Size([100, 4, 50])

    """
    ## Set the device and dtype from the given data tensor
    device = all_representations.device
    dtype = all_representations.dtype


    ## Get the max and min values reported in the rows (0th dimension)
    ## as we vary over all other components (i.e. vary over the embedding dimension)
    maxxes = all_representations.max(0).values
    minns = all_representations.min(0).values
    
    ## DIAGNOSTIC
    if show_diagnostics:
        print(f"minns = {minns}")
        print(f"maxxes = {maxxes}")

    ## Handle online computations -- PRESENTLY UNUSED
    if set_var != 1.0:
        if online_var is not None:
            all_representations = set_var*((all_representations-online_var[1])/(online_var[0]-online_var[1]))
        else:
            #online_var = all_representations.var(0)*(1.0/set_var)
            online_var = (maxxes, minns)
            all_representations = set_var*(all_representations-online_var[1])/(online_var[0]-online_var[1])


    ## Get the desired bins
    if online_bins is not None:
        bins = online_bins
    elif bin_type == "unit_sphere":
        ambient_dim = all_representations.shape[-1]
        bins = get_spherical_bins(n_bins=n_bins, ambient_space_dimension = ambient_dim, device=device, dtype=dtype)
    else:    
        ## This creates the bins by sampling from a space related to the original data 
        ## with a given sampling distribution.
        bins = get_bins2(
            all_representations, 
            bin_type, 
            n_bins, 
    #        n_heads,
#            extra_internal_label_dims_list = extra_internal_label_dims_list,    ## e.g. [n_layers+1, n_heads]
        )
    

    ## DIAGNOSTIC
    if show_diagnostics:
        print()
        print('DIAGNOSTIC in soft_bin2():')
        print(f'all_representations.shape = {all_representations.shape}')
        print(f'bins.shape = {bins.shape}')
        print(f'dist_fn = {dist_fn}')

    ## Compute the soft-binned probability distributions 
#    all_representations = head_reshape(all_representations, n_heads)  ## [N, D] --> [N, n_heads, D//n_heads]
#    scores = distance_scores(all_representations, bins, dist_fn)  ## Returns [N, n_heads, n_bins]
    scores = distance_scores2(all_representations, bins, dist_fn)  ## Returns [N, n_bins]
    scores = smoothing(scores, smoothing_temp, smoothing_fn)   ## Takes / Returns: [N, n_heads, n_bins]

    ## Return the desired output
    return scores, bins


@torch.no_grad()
def soft_bin(all_representations: torch.Tensor,
             n_bins: int,
             bins: Optional[torch.Tensor] = None,
             centers: Optional[torch.Tensor] = None,
             dist_fn: Literal['cosine', 'euclidean', 'dot', 'cosine_5', 'cluster'] = 'euclidean',
             bin_type: Literal['uniform', 'standard_normal', 'unit_sphere', 'unit_cube_by_bins', 'unit_cube_by_interpolation', 'cluster'] = 'uniform',
             sub_mean: bool = False,
             n_heads: int = 1,
             smoothing_fn: Literal["softmax", "sparsemax", "discrete", "None"] = "None",
             smoothing_temp: float = 1.0,
             online_bins: Optional[torch.Tensor] = None,
             set_var: float = 1.0,
             online_var: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             show_diagnostics: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor]:    
    """
    Performs soft binning of neural representations using various distance metrics.

    This is the main function for converting continuous neural representations into
    discrete probability distributions over bins. It supports multiple binning strategies,
    distance functions, and smoothing techniques.

    Args:
        all_representations (torch.Tensor): Input representations to bin [N, D]
        n_bins (int): Number of bins to use for discretization
        bins (torch.Tensor, optional): Pre-computed bin locations
        centers (torch.Tensor, optional): Pre-computed bin centers
        temp (float): Temperature for smoothing (default: 1.0)
        dist_fn (str): Distance function ('cosine', 'euclidean', 'dot')
        bin_type (str): Binning strategy ('uniform', 'unit_sphere', 'cluster', etc.)
        sub_mean (bool): Whether to subtract mean (default: False)
        n_heads (int): Number of attention heads for multi-head processing
        smoothing_fn (str): Smoothing function ('softmax', 'sparsemax', 'discrete')
        online_bins (torch.Tensor, optional): Cached bins from previous iterations
        set_var (float): Variance scaling factor (default: 1.0)
        online_var (tuple, optional): Cached variance statistics

    Returns:
        tuple: (scores, bins)
            - scores: Soft assignment probabilities [N, n_heads, n_bins]
            - bins: Bin locations used for scoring

    Example:
        >>> representations = torch.randn(100, 64)
        >>> scores, bins = soft_bin(representations, n_bins=50)
        >>> print(scores.shape)  # torch.Size([100, 4, 50])

    """
    ## Get the max and min values reported in the rows (0th dimension)
    ## as we vary over all other components (i.e. vary over the embedding dimension)
    maxxes = all_representations.max(0).values
    minns = all_representations.min(0).values
    
    ## DIAGNOSTIC
    if show_diagnostics:
        print(f"minns = {minns}")
        print(f"maxxes = {maxxes}")

    ## Handle online computations -- PRESENTLY UNUSED
    if set_var != 1.0:
        if online_var is not None:
            all_representations = set_var*((all_representations-online_var[1])/(online_var[0]-online_var[1]))
        else:
            #online_var = all_representations.var(0)*(1.0/set_var)
            online_var = (maxxes, minns)
            all_representations = set_var*(all_representations-online_var[1])/(online_var[0]-online_var[1])
    
    ## This creates the bins by sampling from a space related to the original data 
    ## with a given sampling distribution.
    bins = get_bins(
        all_representations, 
        bin_type, 
        n_bins, 
        n_heads
    ) if online_bins is None else online_bins
    

    ## Compute the soft-binned probability distributions 
    all_representations = head_reshape(all_representations, n_heads)  ## [N, D] --> [N, n_heads, D//n_heads]
    scores = distance_scores(all_representations, bins, dist_fn)  ## Returns [N, n_heads, n_bins]
    scores = smoothing(scores, smoothing_temp, smoothing_fn)   ## Takes / Returns: [N, n_heads, n_bins]

    ## Return the desired output
    return scores, bins




@torch.no_grad()
def head_reshape(all_representations: torch.Tensor, n_heads:int) -> torch.Tensor:
    """
    Reshapes representations for multi-head attention-style processing.

    Args:
        all_representations (torch.Tensor): Input tensor [N, D]
        n_heads (int): Number of heads to split into

    Returns:
        torch.Tensor: Reshaped tensor [N, n_heads, D//n_heads]

    Note:
        Assumes that D is divisible by n_heads

    """
    ## Reshapes the tensor so the embedded dimension is split across a 
    ## "head" index and a head-dependent embedded dimension index.
    d_hidden = all_representations.shape[-1]    
    return all_representations.view(-1, n_heads, int(d_hidden/n_heads))



@torch.no_grad()
def get_bins(all_representations:torch.Tensor, 
             bin_type:str, n_bins:int, n_heads:int) -> torch.Tensor:
    """
    Generates bin locations according to the specified strategy.

    This function supports multiple binning strategies for discretizing the representation space:
    - 'uniform': Uniformly random bins within data range
    - 'standard_normal': Standard normal random bins
    - 'unit_sphere': L2-normalized random bins
    - 'unit_cube_by_bins': Evenly spaced bins within data range
    - 'unit_cube_by_interpolation': Evenly spaced interpolation points within data range
    - 'cluster': K-means clustering-based bins

    Args:
        all_representations (torch.Tensor): Input representations [N, D]
        bin_type (str): Binning strategy name
        n_bins (int): Number of bins to generate
        n_heads (int): Number of attention heads for reshaping

    Returns:
        torch.Tensor: Bin locations [n_bins, n_heads, D//n_heads]

    Raises:
        NotImplementedError: If bin_type is not supported

    Example:
        >>> data = torch.randn(100, 64)
        >>> bins = get_bins(data, 'uniform', n_bins=50, n_heads=4)
        >>> print(bins.shape)  # torch.Size([50, 4, 16])

    """    
    ## Get the embedding dimension -- the last dimension of our 2D matrix
    ## type(d_hidden) = int
    d_hidden = all_representations.shape[-1]

    ## Get the max and min value for each embedding dimension -- 1D tensor
    ## rep_*.shape = (d_hidden)
    rep_min, rep_max = all_representations.min(0).values, all_representations.max(0).values
    
    
    if bin_type == 'uniform':
        distribution = Uniform(rep_min, rep_max)
        ## Sample the uniform distribution to get the number of bins
        ## bins.shape = (n_bins, d_hidden) then (n_bins, n_heads, d_hidden/n_heads)
        bins = distribution.sample([n_bins])
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'standard_normal':
        bins = torch.randn((n_bins, d_hidden))
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'unit_sphere':
        bins = torch.randn((n_bins, d_hidden))
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        bins = F.normalize(bins, dim=-1)
       
    elif bin_type == 'unit_cube_by_bins':
        bins = unit_cube_bins(
            start=all_representations.min(0).values,
            stop=all_representations.max(0).values,
            n_bins=n_bins
        ).T
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'unit_cube_by_interpolation':
        bins = interpolate_tensors(
            minns=all_representations.min(0).values,
            maxxes=all_representations.max(0).values,
            steps=n_bins - 1
        ).T
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == "cluster":
        bins  = cluster(
            all_representations, n_bins,
        )
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
    else:
        raise NotImplementedError

    # Ensure bins match the input data's device AND dtype
    bins = bins.to(device=all_representations.device, dtype=all_representations.dtype)
    return bins
    


## DEVELOPER NOTES:
## ----------------
## get_spherical_bins(n_bins:int) -- only needs to know the dimension of the ambient space of the unit sphere, not the data distribution
## get_rectangular_bins(rect_min:tuple, rect_max:tuple, n_bins:int) -- needs to know the two extremal corners of the bounding box
## get_..._bins() --TBD
##
## These should replace the get_bins2() routine, not use any internal label dimensions, and be cleaner/simpler! =)
##

@torch.no_grad()
def get_spherical_bins(n_bins:int, ambient_space_dimension:int, device:torch.device, dtype:torch.dtype) -> torch.Tensor:
    """
    Generates a tensor whose rows are the desired bin locations 
    on the unit sphere ||x|| = 1 in the ambient euclidean space.
    """
    ## Generate the desired number of bins on the unit sphere in the desired ambient space 
    bins = torch.randn((n_bins, ambient_space_dimension))
    bins = F.normalize(bins, dim=-1)

    # Ensure bins match the input data's device AND dtype
    bins = bins.to(device=device, dtype=dtype)
    return bins





@torch.no_grad()
def get_bins2(all_representations:torch.Tensor, 
             bin_type:str, n_bins:int) -> torch.Tensor:
    """
    Generates a tensor whose rows are the desired bin locations 
    by using the desired strategy, given a tensor whose rows are 
    the data points we start with.

    NOTE: Perhaps we should separate this into two get bins routines -- a spherical one, and a Euclidean one?

    
##    Generates bin locations according to the specified strategy.

    This function supports multiple binning strategies for discretizing the representation space:
    - 'uniform': Uniformly random bins within data range
    - 'standard_normal': Standard normal random bins
    - 'unit_sphere': L2-normalized random bins
    - 'unit_cube_by_bins': Evenly spaced bins within data range
    - 'unit_cube_by_interpolation': Evenly spaced interpolation points within data range
    - 'cluster': K-means clustering-based bins

    Args:
        all_representations (torch.Tensor): Input representations [N, D]
        bin_type (str): Binning strategy name
        n_bins (int): Number of bins to generate
##        n_heads (int): Number of attention heads for reshaping

    Returns:
        torch.Tensor: Bin locations [n_bins, D]

    Raises:
        NotImplementedError: If bin_type is not supported

    Example:
        >>> data = torch.randn(100, 64)
        >>> bins = get_bins(data, 'uniform', n_bins=50)
        >>> print(bins.shape)  # torch.Size([50, 64])

    """    
    ## Get the embedding dimension -- the last dimension of our 2D matrix
    ## type(d_hidden) = int
    d_hidden = all_representations.shape[-1]

    ## Get the max and min value for each embedding dimension -- 1D tensor
    ## rep_*.shape = (d_hidden)
    rep_min, rep_max = all_representations.min(0).values, all_representations.max(0).values
    
    
    if bin_type == 'uniform':
        distribution = Uniform(rep_min, rep_max)
        ## Sample the uniform distribution to get the number of bins
        ## bins.shape = (n_bins, d_hidden) then (n_bins, n_heads, d_hidden/n_heads)
        bins = distribution.sample([n_bins])
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'standard_normal':
        bins = torch.randn((n_bins, d_hidden))
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'unit_sphere':
        bins = torch.randn((n_bins, d_hidden))
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        bins = F.normalize(bins, dim=-1)
       
    elif bin_type == 'unit_cube_by_bins':
        bins = unit_cube_bins(
            start=all_representations.min(0).values,
            stop=all_representations.max(0).values,
            n_bins=n_bins
        ).T
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == 'unit_cube_by_interpolation':
        bins = interpolate_tensors(
            minns=all_representations.min(0).values,
            maxxes=all_representations.max(0).values,
            steps=n_bins - 1
        ).T
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == "cluster":
        bins  = cluster(
            all_representations, n_bins,
        )
#        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
    else:
        raise NotImplementedError

    # Ensure bins match the input data's device AND dtype
    bins = bins.to(device=all_representations.device, dtype=all_representations.dtype)
    return bins
    




@torch.no_grad()
def distance_scores(all_representations: torch.Tensor, 
                    bins: torch.Tensor, distance_fn: str) -> torch.Tensor:
    """
    Computes distance scores between representations and bins.

    Supports multiple distance functions for measuring similarity between
    neural representations and discretization bins.

    Args:
        all_representations (torch.Tensor): Input representations [N, n_heads, D//n_heads]
        bins (torch.Tensor): Bin locations [n_bins, n_heads, D//n_heads]
        distance_fn (str): Distance function type
            - 'euclidean': Negative L2 distance
            - 'cosine': Cosine similarity (normalized dot product)
            - 'cosine_5': Scaled cosine similarity (factor of 5)
            - 'dot': Raw dot product
            - 'cluster': Clustering-based scoring

    Returns:
        torch.Tensor: Distance scores [N, n_heads, n_bins]

    Raises:
        NotImplementedError: If distance_fn is not supported

    Note:
        Higher scores indicate greater similarity (closer distance)

    """
    if distance_fn == "euclidean":
        ## This puts the number of heads first, and adds a dummy dimension at the start
        ## [n_bins, n_heads, D//n_heads] -> [1, n_heads, n_bins, D//n_heads]
        bins = bins.permute(1, 0, 2).unsqueeze(0)
        ## This adds a dummy variable in the next-to-last index
        ## [N, n_heads, D//n_heads] -> [N, n_heads, 1, D//n_heads]
        all_representations = all_representations.unsqueeze(-2)
        
        ## This expects the last two dimensions as the number of points and feature dimension
        ## By broadcasting (right-to-left) we have that the shapes
        ##   [1, n_heads, n_bins, D//n_heads]
        ##   [N, n_heads, 1,      D//n_heads]
        ## gives the tensor of euclidean distances of shape
        ##   [N, n_heads, n_bins]
        scores = -torch.cdist(
            all_representations, 
            bins,
            p=2,
        )

        ## Finally, if there is only one head, we remove this (superfluous) index
        ##   [N, n_heads, n_bins]  if n_heads > 1, or
        ##   [N, n_bins]  if n_heads == 1
        scores = scores.squeeze(-2)
        
    elif distance_fn == "cosine":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## Normalize both the data and bins to be on the unit sphere
        all_representations = F.normalize(all_representations, dim=-1)
        bins = F.normalize(bins, dim=-1)
        
        ## This puts the number of bins first, i.e.
        ## [n_bins, n_heads, D//n_heads] -> [n_heads, D//n_heads, n_bins]
        bins = bins.permute(1, 2, 0)

        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, n_heads, D//n_heads]
        ##   bins =                [n_heads, D//n_heads, n_bins]
        ## giving the final dot-product tensor of scores the shape
        ##   [N, n_heads, n_bins]
        scores = torch.einsum('bhd,hdp->bhp', all_representations, bins)
        
    elif distance_fn == "cosine_5":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## Normalize both the data and bins to be on the sphere of radius r=5
        all_representations = F.normalize(all_representations, dim=-1)*5
        bins = F.normalize(bins, dim=-1)*5
        
        ## This puts the number of bins first, i.e.
        ## [n_bins, n_heads, D//n_heads] -> [n_heads, D//n_heads, n_bins]
        bins = bins.permute(1, 2, 0)

        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, n_heads, D//n_heads]
        ##   bins =                [n_heads, D//n_heads, n_bins]
        ## giving the final dot-product tensor of scores the shape
        ##   [N, n_heads, n_bins]
        scores = torch.einsum('bhd,hdp->bhp', all_representations, bins)
        
    elif distance_fn == "dot":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## This puts the number of bins first, i.e.
        ## [n_bins, n_heads, D//n_heads] -> [n_heads, D//n_heads, n_bins]
        bins = bins.permute(1, 2, 0)

        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, n_heads, D//n_heads]
        ##   bins =                [n_heads, D//n_heads, n_bins]
        ## giving the final dot-product tensor of scores the shape
        ##   [N, n_heads, n_bins]
        scores = torch.einsum('bhd,hdp->bhp', all_representations, bins)
        
    elif distance_fn == "cluster":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## This puts the number of bins first, i.e.
        ## [n_bins, n_heads, D//n_heads] -> [n_heads, D//n_heads, n_bins]
        bins = bins.permute(1, 2, 0)

        ## This transforms our data back to the "headless" version, i.e. from 
        ##   all_representations = [N, n_heads, D//n_heads]
        ## to 
        ##   all_representations = [N, D]
        ## before performing a K-means clustering on the data
        scores = cluster(all_representations.view(all_representations.shape[0], -1), 100, just_bins=False)
        
    else:
        raise NotImplementedError
        
    return scores
    

## TO DO:
## NOTE: We want to compute the distance tensor using the last dimension of the tensor, 
##       where the bins are given as a [n_bins, D] tensor, and all_representations 
##       is a [n_samples, *structural_label_dims_list, D] tensor.  Here the output is 
##       is a [n_samples, *structural_label_dims_list; n_bins] tensor.


@torch.no_grad()
def distance_scores2(all_representations: torch.Tensor, 
                    bins: torch.Tensor, distance_fn: str) -> torch.Tensor:
    """
    Computes distance scores between representations and bins.

    Supports multiple distance functions for measuring similarity between
    neural representations and discretization bins.

    Args:
##        all_representations (torch.Tensor): Input representations [N, n_heads, D//n_heads]
        all_representations (torch.Tensor): Input representations [N, D]
##        bins (torch.Tensor): Bin locations [n_bins, n_heads, D//n_heads]
        bins (torch.Tensor): Bin locations [n_bins, D]
        distance_fn (str): Distance function type
            - 'euclidean': Negative L2 distance
            - 'cosine': Cosine similarity (normalized dot product)
            - 'cosine_5': Scaled cosine similarity (factor of 5)
            - 'dot': Raw dot product
            - 'cluster': Clustering-based scoring

    Returns:
##        torch.Tensor: Distance scores [N, n_heads, n_bins]
        torch.Tensor: Distance scores [N, n_bins]

    Raises:
        NotImplementedError: If distance_fn is not supported

    Note:
        Higher scores indicate greater similarity (closer distance)

    """
    if distance_fn == "euclidean":
        ## This puts the number of heads first, and adds a dummy dimension at the start
        ## [n_bins, n_heads, D//n_heads] -> [1, n_heads, n_bins, D//n_heads]
##        bins = bins.permute(1, 0, 2).unsqueeze(0)

        ## [n_bins, D] -> [1, n_bins, D]
        bins = bins.unsqueeze(0)

        ## This adds a dummy variable in the next-to-last index
        ## [N, n_heads, D//n_heads] -> [N, n_heads, 1, D//n_heads]
##        all_representations = all_representations.unsqueeze(-2)
        
        ## This adds a dummy variable in the next-to-last index
        ## [N, D] -> [N, 1, D]
        all_representations = all_representations.unsqueeze(-2)


        ## This expects the last two dimensions as the number of points and feature dimension
        ## By broadcasting (right-to-left) we have that the shapes
        ##   [1, n_heads, n_bins, D//n_heads]
        ##   [N, n_heads, 1,      D//n_heads]
        ## gives the tensor of euclidean distances of shape
        ##   [N, n_heads, n_bins]
        scores = -torch.cdist(
            all_representations, 
            bins,
            p=2,
        )

        ## Finally, if there is only one head, we remove this (superfluous) index
        ##   [N, n_heads, n_bins]  if n_heads > 1, or
        ##   [N, n_bins]  if n_heads == 1
        scores = scores.squeeze(-2)
        
    elif distance_fn == "cosine":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## Normalize both the data and bins to be on the unit sphere
        all_representations = F.normalize(all_representations, dim=-1)
        bins = F.normalize(bins, dim=-1)

        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, *extra_internal_label_dims_list, D]
        ##   bins =                [n_bins, D]
        ## where we broadcast across all extra_internal_dimensions -- 
        ## giving the final dot-product tensor of scores the shape [N, *extra_internal_label_dims_list, n_bins]
        scores = torch.einsum('n...d,bd->n...b', all_representations, bins)


    elif distance_fn == "cosine_5":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## Normalize both the data and bins to be on the sphere of radius r=5
        all_representations = F.normalize(all_representations, dim=-1)*5
        bins = F.normalize(bins, dim=-1) * 5

        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, *extra_internal_label_dims_list, D]
        ##   bins =                [n_bins, D]
        ## where we broadcast across all extra_internal_dimensions -- 
        ## giving the final dot-product tensor of scores the shape [N, *extra_internal_label_dims_list, n_bins]
        scores = torch.einsum('n...d,bd->n...b', all_representations, bins)


    elif distance_fn == "dot":
        '''
        batch x heads x dimensions, heads x dimensions x points (bins)
        -> batch x heads x points (bins)
        '''
        ## Take the dot product in the embedding dimension of both tensors 
        ##   all_representations = [N, *extra_internal_label_dims_list, D]
        ##   bins =                [n_bins, D]
        ## where we broadcast across all extra_internal_dimensions -- 
        ## giving the final dot-product tensor of scores the shape [N, *extra_internal_label_dims_list, n_bins]
        scores = torch.einsum('n...d,bd->n...b', all_representations, bins)

#    _ = """
#    elif distance_fn == "cluster":
#        '''
#        batch x heads x dimensions, heads x dimensions x points (bins)
#        -> batch x heads x points (bins)
#        '''
#        ## This puts the number of bins first, i.e.
#        ## [n_bins, n_heads, D//n_heads] -> [n_heads, D//n_heads, n_bins]
#        bins = bins.permute(1, 2, 0)
#
#        ## This transforms our data back to the "headless" version, i.e. from 
#        ##   all_representations = [N, n_heads, D//n_heads]
#        ## to 
#        ##   all_representations = [N, D]
#        ## before performing a K-means clustering on the data
#        scores = cluster(all_representations.view(all_representations.shape[0], -1), 100, just_bins=False)
#    """
                
    else:
        raise NotImplementedError
        
    return scores    




## NOTE: This applies smoothing based on normalizations applied in the last dimension!

@torch.no_grad()
def smoothing(scores: torch.Tensor, temp: float, smoothing_fn: str) -> torch.Tensor:
    """
    Applies smoothing function to convert distance scores to probabilities.

    Args:
        scores (torch.Tensor): Raw distance scores [N, n_heads, n_bins]
        temp (float): Temperature parameter for controlling sharpness
        smoothing_fn (str): Smoothing function type
            - 'softmax': Standard softmax (continuous)
            - 'sparsemax': Sparse softmax (can produce exact zeros)
            - 'discrete': Hard assignment (one-hot)
            - 'None': No smoothing (return raw scores)

    Returns:
        torch.Tensor: Smoothed probability distributions [N, n_heads, n_bins]

    Raises:
        NotImplementedError: If smoothing_fn is not supported

    Note:
        Lower temperature values produce sharper (more discrete) distributions
    """    
    if smoothing_fn == "sparsemax":
        scores = sparsemax(scores/temp, dim=-1)
        
    elif smoothing_fn == "softmax":
        scores = F.softmax(scores/temp, dim=-1)
    
    elif smoothing_fn == "discrete":
        ## Chooses the largest component as the one-hot label -- not temperature-dependent
        scores = F.one_hot(scores.argmax(dim=-1), num_classes=scores.size(-1))
        
    elif smoothing_fn == 'None':
        scores = scores
        
    else:
        raise NotImplementedError
        
    return scores



@torch.no_grad()
def cluster(all_representations: torch.Tensor,
            n_bins: int,
            just_bins: bool = True
            ) -> torch.Tensor:
    """
    Performs K-means clustering for bin generation or scoring.

    Args:
        all_representations (torch.Tensor): Input representations [N, D]
        n_bins (int): Number of clusters (bins)
        just_bins (bool): If True, return cluster centers; if False, return assignments

    Returns:
        torch.Tensor: Either cluster centers [n_bins, D] or assignments [N, 1, n_bins]

    Note:
        Requires sklearn.cluster.KMeans (currently commented out)
        Moves data to CPU for clustering, then back to original device
    """
    clustered = KMeans(n_clusters=n_bins).fit(all_representations.to('cpu'))
    centres = torch.tensor(clustered.cluster_centers_).to(all_representations.device).float()
    if just_bins:
        return centres.to(all_representations.device)
    else:
        scores = F.one_hot(
            torch.tensor(clustered.labels_, dtype=torch.long), 
            num_classes=n_bins
        ).unsqueeze(1).to(all_representations.device)
        
        
        return scores




## ==========================================================================
## ===================== LINEAR INTERPOLATION ROUTINES ======================
## ==========================================================================


@torch.no_grad()
@torch.jit.script
def unit_cube_bins(start: torch.Tensor, stop: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Creates evenly spaced bins between start and stop values across multiple dimensions.

    This function replicates multi-dimensional behavior of numpy.linspace in PyTorch
    and is optimized for use with TorchScript compilation.

    Args:
        start (torch.Tensor): Starting values for each dimension
        stop (torch.Tensor): Ending values for each dimension
        n_bins (int): Number of bins to create (will be incremented by 1)

    Returns:
        torch.Tensor: Bin centers with shape [n_dims, n_bins]

    Example:
        >>> start = torch.tensor([0.0, -1.0])
        >>> stop = torch.tensor([1.0, 1.0])
        >>> centers = unit_cube_bins(start, stop, 5)
        >>> print(centers.shape)  # torch.Size([2, 5])

    """
    n_bins +=1
    # create a tensor of 'n_bins' steps from 0 to 1
    steps = torch.arange(n_bins, dtype=torch.float32, device=start.device) / (n_bins - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    bins = (start[None] + steps*(stop - start)[None]).T
    
    bin_widths = (bins[:, 1:] - bins[:, :-1])
    centers = bins[:, :-1] + (bin_widths/2)
        
    return centers



@torch.no_grad()
def interpolate_tensors(minns: torch.Tensor, maxxes: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Create a 2D tensor of coordinate-wise linear interpolations between min and max values.
    
    This function generates evenly spaced interpolated values between corresponding 
    elements of two 1D tensors, creating a 2D output where each row contains the 
    interpolated sequence for one coordinate pair.
    
    Args:
        minns (torch.Tensor): 1D tensor of minimum values for each coordinate.
            Shape: (n,) where n is the number of coordinates.
        maxxes (torch.Tensor): 1D tensor of maximum values for each coordinate.
            Must have the same shape as minns.
        steps (int, optional): Number of interpolation steps (points) to generate
            for each coordinate pair. Defaults to 11.
    
    Returns:
        torch.Tensor: 2D tensor of interpolated values with shape (len(minns), steps).
            Each row i contains `steps` evenly spaced values from minns[i] to maxxes[i].
    
    Example:
        >>> minns = torch.tensor([0.0, 1.0, 2.0])
        >>> maxxes = torch.tensor([10.0, 5.0, 8.0])
        >>> result = interpolate_tensors(minns, maxxes, 5)
        >>> print(result.shape)
        torch.Size([3, 5])
        >>> print(result)
        tensor([[ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000],
                [ 1.0000,  2.0000,  3.0000,  4.0000,  5.0000],
                [ 2.0000,  3.5000,  5.0000,  6.5000,  8.0000]])
    
    Note:
        This function is equivalent to:
        torch.stack([torch.linspace(minns[i], maxxes[i], steps) 
                    for i in range(len(minns))])
        but is more efficient due to vectorized operations and PyTorch's 
        optimized linear interpolation kernel.
    """
    t = torch.linspace(0, 1, steps).unsqueeze(0)
    minns = minns.unsqueeze(1)
    maxxes = maxxes.unsqueeze(1)
    
    return torch.lerp(minns, maxxes, t)




## ==========================================================================
## ==================== MEASUREMENT REPORTING ROUTINES ======================
## ==========================================================================


@torch.no_grad()
def multi_js_divergence(classes: torch.Tensor, p_class: torch.Tensor, 
                        max_normalization: str = "weighted") -> torch.Tensor:
    """
    Computes multi-way Jensen-Shannon divergence.

    This function implements the multi-way generalization of Jensen-Shannon
    divergence for measuring separation between multiple class distributions.

    Args:
        classes (torch.Tensor): Normalized class distributions -- [n_classes, embedding_dim]
        p_class (torch.Tensor): Class prior probabilities -- [n_classes]

    Returns:
        torch.Tensor: Normalized JS divergence score

    Note:
        Result is normalized by the maximum possible entropy to ensure
        scores are comparable across different numbers of classes

    """
    ## Expand p_class for broadcasting:  [n_classes] --> [n_classes, 1]
    p_expanded = p_class.unsqueeze(-1)

    ## Compute mixture distributions: [n_classes, 1] * [n_classes, embedding_dim]  --> [1, embedding_dim]
    m = torch.sum(p_expanded * classes, dim=0)

    ## Compute entropies 
    class_entropies = entropy(classes)  # [n_classes, embedding_dim] --> [n_classes]
    m_entropies = entropy(m)  # [1, embedding_dim] --> []

    ## Compute the weighted average of class entropies: [batch_size]
    weighted_average_class_entropy = torch.sum(p_class * class_entropies, dim=-1)   # [n_classes] * [n_classes] --> []

    ## Compute the JS divergence: [] 
    js_divs = m_entropies - weighted_average_class_entropy   # [] - [] --> []


    ## Normalization bounds (theoretical maximum JS divergence -- uniform and weighted)
    uniform_entropy = torch.log(torch.tensor(p_class.shape[0]))   ## Max at the uniform distribution
    weighted_class_entropy = entropy(p_expanded.T, normalization=None).mean()   ## p_expanded.T ==> [1, n_classes]
    
    ## Compute the desired normalized JS divergence
    allowed_max_normalizations = ["uniform", "weighted"]
    if max_normalization == "uniform":
        result = js_divs / uniform_entropy
    elif max_normalization == "weighted":
        result = js_divs / weighted_class_entropy
    else:
        raise ValueError(f"max_normalization = {max_normalization} must be in {allowed_max_normalizations}.")
    
    ## Return the desired result
    return result





@torch.no_grad()
def js_divergence(p: torch.Tensor, q: torch.Tensor, 
                  eps: float = 1e-9, use_xlogy: bool = True,
                  normalization: str = None) -> torch.Tensor:
    """
    Computes Jensen-Shannon divergence between two distributions.
    
    The JS divergence is symmetric and bounded between 0 and ln(2).
    It measures the similarity between two probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Args:
        p, q (torch.Tensor): Input distributions. Last dimension should contain
                           the probability values.
        normalization (str, optional): How to normalize inputs if the last index values don't sum to 1:
                                     - None: No normalization (assume already normalized)
                                     - "scaling": Divide by sum (L1 normalization)  
                                     - "softmax": Apply softmax normalization
        eps (float): Small epsilon for numerical stability. Default: 1e-9

    Returns:
        torch.Tensor: JS divergence score(s)
        
    Raises:
        ValueError: If the normalization parameter is invalid
        
    Note:
        - The input tensors must have the same shape or be broadcastable
        - Normalization is applied along the last dimension
        - For numerical stability when use_xlogy is False, a small epsilon 
            (clamping) parameter is present to address issues at log(0).
    """
    ## Validate the normalization parameter
    valid_normalizations = {None, "scaling", "softmax"}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. "
                        f"Must be one of {valid_normalizations}")

    ## Normalize the distribution in case it doesn't already sum to 1
    if normalization is not None:
        if normalization == "scaling":
            p = normalize_by_scaling(p)
            q = normalize_by_scaling(q)
        elif normalization == "softmax":
            p = normalize_by_softmax(p)
            q = normalize_by_softmax(q)

    ## Compute mean / mixture distribution  m = 0.5 * (p + q)
    m = 0.5 * (p + q)

    ## Compute the average of the two KL-divergences with m
    js_div = 0.5 * kl_divergence(p, m, normalization=None, eps=eps, use_xlogy=use_xlogy) + \
             0.5 * kl_divergence(q, m, normalization=None, eps=eps, use_xlogy=use_xlogy)
    
    ## Return the desired value
    return js_div




@torch.no_grad()
def kl_divergence(p: torch.Tensor, q: torch.Tensor, 
                  eps: float = 1e-9, use_xlogy: bool = True, 
                  normalization: str = None) -> torch.Tensor:
    """
    Computes Kullback-Leibler divergence between distributions.

    Args:
        p (torch.Tensor): First distribution (typically empirical)
        q (torch.Tensor): Second distribution (typically reference)

    Returns:
        torch.Tensor: KL divergence D(p||q)

    Note:
        Uses clamping to avoid numerical issues with log(0)

    """
    ## Validate the normalization parameter
    valid_normalizations = {None, "scaling", "softmax"}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. "
                        f"Must be one of {valid_normalizations}")

    ## Normalize the distribution in case it doesn't already sum to 1
    if normalization is not None:
        if normalization == "scaling":
            p = normalize_by_scaling(p)
            q = normalize_by_scaling(q)
        elif normalization == "softmax":
            p = normalize_by_softmax(p)
            q = normalize_by_softmax(q)


    # Method 1: Use xlogy if available and precision is acceptable
    if use_xlogy and hasattr(torch, 'xlogy'):
        return torch.xlogy(p, p / q.clamp(min=eps)).sum(-1)
    
    # Method 2: Fallback to torch.where if higher precision is desired
    q_safe = q.clamp(min=eps)
    p_safe = p.clamp(min=eps)  
    log_ratio = p_safe.log() - q_safe.log()
    return torch.where(p > 0, p * log_ratio, torch.tensor(0.0, device=p.device)).sum(-1)





@torch.no_grad()
@torch.compile
def normalize_by_scaling(dist: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normalizes distributions to sum to 1 along the last dimension by 
    scaling the entries by the inverse of their sum.  This returns a 
    tensor of the same shape as the input tensor dist.

    Args:
        dist (torch.Tensor): Unnormalized distribution

    Returns:
        torch.Tensor: Normalized distribution

    Note:
        Uses clamping to avoid division by zero

    """
    return dist / dist.sum(-1, keepdim=True).clamp(min=eps)





@torch.no_grad()
def normalize_by_softmax(dist: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Normalizes distributions to sum to 1 along the last dimension by 
    applying the softmax function.  This returns a tensor of the same shape 
    as the input tensor dist.

    Args:
        dist (torch.Tensor): Unnormalized distribution

    Returns:
        torch.Tensor: Normalized distribution

    """
    return torch.softmax(dist / temperature, dim=-1)




@torch.no_grad()
def entropy(dist: torch.Tensor, normalization: str = None) -> float:
    """
    Computes Shannon entropy of distributions.

    Args:
        dist (torch.Tensor): Probability distribution(s)

    Returns:
        torch.Tensor: Entropy values

    Formula:
        H(p) = -∑ p(x) log p(x)

    Note:
        Automatically normalizes input distributions and uses clamping
        to handle numerical issues with log(0)

    """
    ## Validate the normalization parameter
    valid_normalizations = {None, "scaling", "softmax"}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. "
                        f"Must be one of {valid_normalizations}")

    ## Normalize the distribution in case it doesn't already sum to 1
    if normalization is not None:
        if normalization == "scaling":
            p = normalize_by_scaling(dist)
        elif normalization == "softmax":
            p = normalize_by_softmax(dist)
    else:
        p = dist


    ## Compute the entropy
    ## -------------------
    # Method 1: Compute entropy with the pytorch function entr if available (PyTorch 1.7+)
    if hasattr(torch.special, 'entr'):
        return torch.special.entr(p).sum(-1)
    
    # Method 2: Fallback to with torch.where (unclamped) otherwise 
    log_p = p.log()
    return torch.where(p > 0, -p * log_p, torch.tensor(0.0, device=p.device)).sum(-1)

    



## ==========================================================================
## ==================== ONLINE ENTROPY ACCUMULATOR ==========================
## ==========================================================================


class EntropyAccumulator:
    """
    Stateful accumulator for online/distributed entropy calculations.

    Maintains running counts and bins for incremental entropy computation without
    needing to store all data. Supports distributed computation via merge().

    Example:
        >>> # Online computation
        >>> acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])
        >>> for batch_data, batch_labels in dataloader:
        >>>     acc.update(batch_data, batch_labels)
        >>> metrics = acc.compute_metrics()

        >>> # Distributed computation
        >>> acc1 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        >>> acc2 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        >>> acc1.update(batch1_data, batch1_labels)
        >>> acc2.update(batch2_data, batch2_labels)
        >>> acc1.merge(acc2)
        >>> metrics = acc1.compute_metrics()
    """

    def __init__(self, n_bins: int, label_list: list, embedding_dim: Optional[int] = None,
                 n_heads: int = 1, 
                 dist_fn: str = 'euclidean', bin_type: str = 'uniform',
                 smoothing_fn: str = 'None', smoothing_temp: float = 1.0):
        """
        Initialize the entropy accumulator.

        Args:
            n_bins: Number of bins for soft-binning
            label_list: List of unique label names
            embedding_dim: Dimension of data embeddings. Required for data-independent bin types
                          like 'unit_sphere', 'standard_normal'. Optional for data-dependent
                          bin types like 'uniform'. If provided, bins are pre-computed.
            #n_heads: Number of attention heads (default: 1)
            extra_label_dim_size_vector (default = [])
            dist_fn: Distance function for soft-binning (default: 'euclidean')
            bin_type: Binning strategy (default: 'uniform')
            smoothing_fn: Smoothing function (default: 'None')
            smoothing_temp: Temperature for smoothing (default: 1.0)
        """
        self.n_bins = n_bins
        self.label_list = label_list
        self.embedding_dim = embedding_dim
#        self.n_heads = n_heads
        self.n_heads = 1               ## DELETE THIS AFTER DEPRECATING IT!
        self.extra_label_dim_size_vector = []
        self.num_labels = len(label_list)
        self.dist_fn = dist_fn
        self.bin_type = bin_type
        self.smoothing_fn = smoothing_fn
        self.smoothing_temp = smoothing_temp

        # Running counts (unnormalized) - initialized on first update
        self.total_count = 0
        self.total_scores_sum = None      # Shape: [n_bins]
        self.label_scores_sum = None      # Shape: [num_labels, n_bins]
        self.label_counts = None          # Shape: [num_labels]

        # Fixed bins - pre-compute if embedding_dim is provided and bin_type is data-independent
        self.bins = None
        self.dtype = None
        self.device = None

        # Pre-compute bins if possible
        if embedding_dim is not None and bin_type in ['unit_sphere', 'standard_normal']:
            self._precompute_bins()


    @torch.no_grad()
    def _precompute_bins(self):
        """
        Pre-compute bins for data-independent bin types.
        Only works for 'unit_sphere' and 'standard_normal' bin types.
        """
        d_hidden = self.embedding_dim

        if self.bin_type == 'unit_sphere':
            bins = torch.randn((self.n_bins, d_hidden))
            bins = bins.view(self.n_bins, self.n_heads, int(d_hidden/self.n_heads))
            bins = F.normalize(bins, dim=-1)
        elif self.bin_type == 'standard_normal':
            bins = torch.randn((self.n_bins, d_hidden))
            bins = bins.view(self.n_bins, self.n_heads, int(d_hidden/self.n_heads))
        else:
            raise ValueError(f"Cannot pre-compute bins for bin_type='{self.bin_type}'. "
                           f"Only 'unit_sphere' and 'standard_normal' are supported.")

        self.bins = bins
        # Note: dtype and device will be set when first batch is processed

    @torch.no_grad()
    def update(self, data_tensor: torch.Tensor, index_tensor: torch.Tensor):
        """
        Add a new batch of data to the accumulator.

        Args:
            data_tensor: Data embeddings [N, D]
            index_tensor: Label indices [N], values in range [0, num_labels)
        """
        # Initialize dtype and device on first update
        if self.dtype is None:
            self.dtype = data_tensor.dtype
            self.device = data_tensor.device

        # Compute soft-bin scores using existing or pre-computed bins
        if self.bins is None:
            # No pre-computed bins - compute from data (for data-dependent bin types)
            scores, self.bins = soft_bin(
                data_tensor,
                n_bins=self.n_bins,
                n_heads=self.n_heads,
                dist_fn=self.dist_fn,
                bin_type=self.bin_type,
                smoothing_fn=self.smoothing_fn,
                smoothing_temp=self.smoothing_temp
            )
            # Ensure bins match data dtype for consistency
            self.bins = self.bins.to(dtype=data_tensor.dtype, device=data_tensor.device)
        else:
            # Bins exist (either pre-computed or from first batch) - reuse them
            bins_same_dtype = self.bins.to(dtype=data_tensor.dtype, device=data_tensor.device)
            scores, _ = soft_bin(
                data_tensor,
                n_bins=self.n_bins,
                n_heads=self.n_heads,
                online_bins=bins_same_dtype,  # Use existing bins with matching dtype
                dist_fn=self.dist_fn,
                bin_type=self.bin_type,
                smoothing_fn=self.smoothing_fn,
                smoothing_temp=self.smoothing_temp
            )

        # Remove heads dimension
        scores_no_heads = scores.squeeze(1)

        # Initialize accumulators on first call
        if self.total_scores_sum is None:
            self.total_scores_sum = torch.zeros(self.n_bins, dtype=self.dtype, device=self.device)
            self.label_scores_sum = torch.zeros(self.num_labels, self.n_bins, dtype=self.dtype, device=self.device)
            self.label_counts = torch.zeros(self.num_labels, dtype=torch.long, device=self.device)

        # Accumulate counts
        self.total_count += scores_no_heads.shape[0]
        self.total_scores_sum += scores_no_heads.sum(0)
        self.label_scores_sum.index_add_(dim=0, source=scores_no_heads, index=index_tensor)
        self.label_counts += torch.bincount(index_tensor, minlength=self.num_labels)

    @torch.no_grad()
    def compute_metrics(self, conditional_entropy_label_weighting: Literal["weighted", "uniform"] = "weighted") -> dict:
        """
        Compute current entropy metrics from accumulated state.

        Args:
            conditional_entropy_label_weighting: Weighting scheme for conditional entropy
                - "weighted": Weight by empirical label probabilities
                - "uniform": Weight all labels equally

        Returns:
            Dictionary containing:
                - entropy: Total population entropy
                - conditional_entropy: Conditional entropy H(Z|L)
                - mutual_information: Mutual information I(Z;L)
                - label_entropy_dict: Per-label and total population entropies
                - intermediate_data: Probability distributions and bins
        """
        if self.total_count == 0:
            raise ValueError("Cannot compute metrics on empty accumulator. Call update() first.")

        # Normalize to get probability distributions
        prob_total = self.total_scores_sum / self.total_count
        prob_by_label = self.label_scores_sum / self.label_counts.unsqueeze(1)
        label_distribution = (self.label_counts / self.total_count).to(torch.float64)

        # Compute entropies
        total_entropy = entropy(prob_total)
        entropy_by_label = entropy(prob_by_label).to(torch.float64)

        # Build entropy dictionary
        entropy_dict = {'total_population': total_entropy.item()}
        for i, label in enumerate(self.label_list):
            entropy_dict[label] = entropy_by_label[i].item()

        # Conditional entropy
        if conditional_entropy_label_weighting == "weighted":
            cond_entropy = torch.dot(label_distribution, entropy_by_label).item()
        else:  # uniform
            n = len(self.label_list)
            uniform_dist = torch.ones(n, device=self.device) / n
            cond_entropy = torch.dot(uniform_dist, entropy_by_label).item()

        # Mutual information
        mutual_info = total_entropy.item() - cond_entropy

        return {
            'output_metrics': {
                'entropy': total_entropy.item(),
                'conditional_entropy': cond_entropy,
                'mutual_information': mutual_info,
                'label_entropy_dict': entropy_dict,
            },
            'intermediate_data': {
                'prob_dist_for_total_population_tensor': prob_total,
                'prob_dist_by_label_tensor': prob_by_label,
                'tmp_bins': self.bins,
            }
        }

    @torch.no_grad()
    def merge(self, other: 'EntropyAccumulator'):
        """
        Merge state from another accumulator for distributed computation.

        Args:
            other: Another EntropyAccumulator to merge into this one

        Raises:
            ValueError: If accumulators have incompatible configurations
        """
        # Validate compatibility
        if self.n_bins != other.n_bins:
            raise ValueError(f"Cannot merge: n_bins mismatch ({self.n_bins} != {other.n_bins})")
        if self.num_labels != other.num_labels:
            raise ValueError(f"Cannot merge: num_labels mismatch ({self.num_labels} != {other.num_labels})")
        if self.label_list != other.label_list:
            raise ValueError(f"Cannot merge: label_list mismatch")

        # If this accumulator is empty, adopt other's bins
        if self.bins is None:
            self.bins = other.bins
            self.dtype = other.dtype
            self.device = other.device
            self.total_scores_sum = other.total_scores_sum.clone() if other.total_scores_sum is not None else None
            self.label_scores_sum = other.label_scores_sum.clone() if other.label_scores_sum is not None else None
            self.label_counts = other.label_counts.clone() if other.label_counts is not None else None
            self.total_count = other.total_count
            return

        # If other is empty, nothing to merge
        if other.bins is None:
            return

        # Both have data - merge counts
        self.total_count += other.total_count
        self.total_scores_sum += other.total_scores_sum
        self.label_scores_sum += other.label_scores_sum
        self.label_counts += other.label_counts

    def get_state_dict(self) -> dict:
        """
        Get serializable state dictionary for saving/loading.

        Returns:
            Dictionary containing all accumulator state
        """
        return {
            'n_bins': self.n_bins,
            'label_list': self.label_list,
            'embedding_dim': self.embedding_dim,
            'n_heads': self.n_heads,
            'num_labels': self.num_labels,
            'dist_fn': self.dist_fn,
            'bin_type': self.bin_type,
            'smoothing_fn': self.smoothing_fn,
            'smoothing_temp': self.smoothing_temp,
            'total_count': self.total_count,
            'total_scores_sum': self.total_scores_sum,
            'label_scores_sum': self.label_scores_sum,
            'label_counts': self.label_counts,
            'bins': self.bins,
            'dtype': self.dtype,
            'device': str(self.device) if self.device is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> 'EntropyAccumulator':
        """
        Restore accumulator from state dictionary.

        Args:
            state_dict: Dictionary from get_state_dict()

        Returns:
            Restored EntropyAccumulator instance
        """
        acc = cls(
            n_bins=state_dict['n_bins'],
            label_list=state_dict['label_list'],
            embedding_dim=state_dict.get('embedding_dim'),  # Use .get() for backward compatibility
            n_heads=state_dict['n_heads'],
            dist_fn=state_dict['dist_fn'],
            bin_type=state_dict['bin_type'],
            smoothing_fn=state_dict['smoothing_fn'],
            smoothing_temp=state_dict['smoothing_temp']
        )
        acc.num_labels = state_dict['num_labels']
        acc.total_count = state_dict['total_count']
        acc.total_scores_sum = state_dict['total_scores_sum']
        acc.label_scores_sum = state_dict['label_scores_sum']
        acc.label_counts = state_dict['label_counts']
        acc.bins = state_dict['bins']
        acc.dtype = state_dict['dtype']
        acc.device = torch.device(state_dict['device']) if state_dict['device'] is not None else None
        return acc




## ==========================================================================
## ==================== ONLINE ENTROPY ACCUMULATOR -- REVISED VERSION ==========================
## ==========================================================================


class EntropyAccumulator2():
    """
    Stateful accumulator for online/distributed entropy calculations.

    Maintains running counts and bins for incremental entropy computation without
    needing to store all data. Supports distributed computation via merge().

    Example:
        >>> # Online computation
        >>> acc = EntropyAccumulator(n_bins=10, label_list=['A', 'B', 'C'])
        >>> for batch_data, batch_labels in dataloader:
        >>>     acc.update(batch_data, batch_labels)
        >>> metrics = acc.compute_metrics()

        >>> # Distributed computation
        >>> acc1 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        >>> acc2 = EntropyAccumulator(n_bins=10, label_list=['A', 'B'])
        >>> acc1.update(batch1_data, batch1_labels)
        >>> acc2.update(batch2_data, batch2_labels)
        >>> acc1.merge(acc2)
        >>> metrics = acc1.compute_metrics()
    """

    _ = '''
    def __init__(self, n_bins: int, label_name: str = 'label', 
                 initial_label_list: list = [], 
                 embedding_dim: Optional[int] = None,
#                 n_heads: int = 1, 
                 extra_internal_label_dims_list = [],
                 extra_internal_label_dims_name_list = [],
#                 extra_label_dim_size_vector: list[int] = [], 
                 probability_label_dim_name = 'probability_label',
                 dist_fn: str = 'euclidean', bin_type: str = 'uniform',
                 smoothing_fn: str = 'None', smoothing_temp: float = 1.0):
       """
        Initialize the entropy accumulator.

        Args:
            n_bins: Number of bins for soft-binning
            label_name: String for the name of the label
            initial_label_list: List of unique label names (which can be added to by our update() routine)
            embedding_dim: Dimension of data embeddings. Required for data-independent bin types
                          like 'unit_sphere', 'standard_normal'. Optional for data-dependent
                          bin types like 'uniform'. If provided, bins are pre-computed.
            #n_heads: Number of attention heads (default: 1)
            extra_label_dim_size_vector (default = [])
            dist_fn: Distance function for soft-binning (default: 'euclidean')
            bin_type: Binning strategy (default: 'uniform')
            smoothing_fn: Smoothing function (default: 'None')
            smoothing_temp: Temperature for smoothing (default: 1.0)
        """
    '''

    
    def __init__(self, config: Optional[EntropyEstimatorConfig] = None):
 
        """
        Initialize the entropy accumulator.

        Args:
            config: Configuration for the estimator. If None, uses defaults.
        """
        ## DIAGNOSTIC
        print("Starting ___init__()")


        # Use provided config or create default
        self.config = config or EntropyEstimatorConfig()
        
        # Extract all parameters from config
        self.n_bins = self.config.n_bins
        self.n_heads = self.config.n_heads
        self.bin_type = self.config.bin_type
        self.dist_fn = self.config.dist_fn
        self.smoothing_fn = self.config.smoothing_fn
        self.smoothing_temp = self.config.smoothing_temp
        
        self.label_name = self.config.label_name
        self.label_list = self.config.initial_label_list
        self.n_labels = len(self.label_list)
        self.probability_label_dim_name = self.config.probability_label_dim_name
        
        self.embedding_dim = self.config.embedding_dim
        self.extra_internal_label_dims_list = self.config.extra_internal_label_dims_list
        self.extra_internal_label_dims_name_list = self.config.extra_internal_label_dims_name_list
     

        # Running counts (unnormalized) - initialized on first update
        self.label_counts = None          # Shape: [num_labels]
        self.granular_label_scores_sum = None     # Shape: [num_labels, *extra_internal_label_dims_list, n_bins]



        # Fixed bins - pre-compute if embedding_dim is provided and bin_type is data-independent
        self.bins = None
        self.dtype = None
        self.device = None


        # Pre-compute bins if possible
        if self.embedding_dim is not None and self.bin_type in ['unit_sphere', 'standard_normal']:
            self._precompute_bins()

        ## DIAGNOSTIC
        print("Finishing ___init__()")




    def __repr__(self):
        ## Define the output string
        out_str = "EntropyAccumulator2 instance"

        ## Return the desired output string
        return out_str





    @torch.no_grad()
    def _precompute_bins(self):
        """
        Pre-compute bins for data-independent bin types.
        Only works for 'unit_sphere' and 'standard_normal' bin types.
        """
        ## DIAGNOSTIC
        print("Starting _precompute_bins()")

        ## Alias the hidden dimension
        d_hidden = self.embedding_dim

        if self.bin_type == 'unit_sphere':
            ## Choose a consistent set of bins in the embedding dimension
            bins = torch.randn((self.n_bins, d_hidden))
#            bins = bins.view(self.n_bins, self.n_heads, int(d_hidden/self.n_heads))
            bins = F.normalize(bins, dim=-1)
        elif self.bin_type == 'standard_normal':
            bins = torch.randn((self.n_bins, d_hidden))
#            bins = bins.view(self.n_bins, self.n_heads, int(d_hidden/self.n_heads))
        else:
            raise ValueError(f"Cannot pre-compute bins for bin_type='{self.bin_type}'. "
                           f"Only 'unit_sphere' and 'standard_normal' are supported.")

        self.bins = bins
        # Note: dtype and device will be set when first batch is processed

        ## DIAGNOSTIC
        print("Finishing _precompute_bins()")




    @torch.no_grad()
    def update(self, data_tensor: torch.Tensor, batch_index_tensor: torch.Tensor, batch_label_list: list, SHOW_DIAGNOSTICS=False):
        """
        Add a new batch of data to the accumulator.

        Args:
            data_tensor (torch.Tensor): Data embeddings of shape [N, *extra_internal_label_dims_list, D]
                where N is the number of samples, D is the embedding dimension, and
                extra_internal_label_dims_list contains any intermediate dimensions (e.g., layers, heads).
                Dtype should be float32 or float64.

            batch_index_tensor (torch.Tensor): Label indices of shape [N] with integer dtype (e.g., torch.long).
                Expected values:
                  - Non-negative integers in the contiguous range [0, len(batch_label_list)-1] for valid samples.
                    Each index i maps to the label at batch_label_list[i].
                  - -1 (or any negative value) for samples that should be ignored/masked
                    (e.g., padding tokens). These masked samples will be excluded from accumulation.
                Note: The valid indices should form a contiguous range starting from 0. All indices
                in [0, len(batch_label_list)-1] must have a corresponding entry in batch_label_list,
                though not every index needs to appear in batch_index_tensor.

            batch_label_list (list): List of unique label names/identifiers for this batch.
                The i-th element is the label name corresponding to index i in batch_index_tensor.
                Typically constructed as list(set(labels_in_batch)), with indices assigned via enumerate().
                Labels not previously seen will be added to the accumulator's global label list.

            SHOW_DIAGNOSTICS (bool): If True, print diagnostic information during processing.

        Note:
            Masking support: Negative values in batch_index_tensor (typically -1) indicate
            samples to ignore. This is useful for excluding padding tokens or other invalid
            positions from the entropy accumulation. These samples are filtered out before
            updating granular_label_scores_sum and label_counts.
        """
        ## SANITY CHECK: Are the batch labels unique?
        if len(batch_label_list) != len(set(batch_label_list)):
            raise RuntimeError(f"The labels in batch_label_list = {batch_label_list} are not unique!")

        ## Define the new accumulator label list
        new_accumulator_label_list = self.label_list + [x  for x in batch_label_list  if x not in self.label_list]

        ## Define a mapping from the given batch label list to the label list for the accumulator
        batch_label_index_to_accumulator_label_index_dict = {i: new_accumulator_label_list.index(label_i)
                                                               for i, label_i in enumerate(batch_label_list)}

        ## Create an associated lookup tensor
        max_key = max(batch_label_index_to_accumulator_label_index_dict.keys())
        lookup = torch.zeros(max_key + 1, dtype=batch_index_tensor.dtype)
        for k, v in batch_label_index_to_accumulator_label_index_dict.items():
            lookup[k] = v

        ## Make a new index_tensor for the accumulator label indices -- apply the mapping via indexing.
        ## NOTE: We must handle negative indices (e.g., -1 for padding/ignored tokens) carefully.
        ## PyTorch negative indexing would wrap around (lookup[-1] = lookup[max_key]), corrupting the mapping.
        ## Instead, we only remap valid (non-negative) indices and preserve negative values as-is.
        valid_batch_mask = batch_index_tensor >= 0
        index_tensor = torch.full_like(batch_index_tensor, -1)  # Initialize with -1 (invalid/masked)
        index_tensor[valid_batch_mask] = lookup[batch_index_tensor[valid_batch_mask]]

        ## Update the label list for the accumulator
        self.label_list = new_accumulator_label_list
        self.n_labels = len(new_accumulator_label_list)



        # Initialize dtype and device on first update
        if self.dtype is None:
            self.dtype = data_tensor.dtype
            self.device = data_tensor.device


        ## DIAGNOSTIC
        if SHOW_DIAGNOSTICS:
            print()
            print("DIAGNOSTIC:")
            print("-----------")
            print(f"data_tensor.shape = {data_tensor.shape}")
            print(f"self.n_bins = {self.n_bins}")
            print(f"self.extra_internal_label_dims_list = {self.extra_internal_label_dims_list}")
            print(f"self.dist_fn = {self.dist_fn}")
            print(f"self.bin_type = {self.bin_type}")
            print(f"self.smoothing_fn = {self.smoothing_fn}")
            print(f"self.smoothing_temp = {self.smoothing_temp}")
            #print(f" = {}")
            #print(f" = {}")
            print()



        # Compute soft-bin scores using existing or pre-computed bins
        if self.bins is None:
            # No pre-computed bins - compute from data (for data-dependent bin types)
            scores, self.bins = soft_bin2(
                data_tensor,
                n_bins=self.n_bins,
#                n_heads=self.n_heads,
                extra_internal_label_dims_list=self.extra_internal_label_dims_list,
                dist_fn=self.dist_fn,
                bin_type=self.bin_type,
                smoothing_fn=self.smoothing_fn,
                smoothing_temp=self.smoothing_temp
            )
            # Ensure bins match data dtype for consistency
            self.bins = self.bins.to(dtype=data_tensor.dtype, device=data_tensor.device)
        else:
            # Bins exist (either pre-computed or from first batch) - reuse them
            bins_same_dtype = self.bins.to(dtype=data_tensor.dtype, device=data_tensor.device)
            scores, _ = soft_bin2(
                data_tensor,
                n_bins=self.n_bins,
#                n_heads=self.n_heads,
                extra_internal_label_dims_list=self.extra_internal_label_dims_list,
                online_bins=bins_same_dtype,  # Use existing bins with matching dtype
                dist_fn=self.dist_fn,
                bin_type=self.bin_type,
                smoothing_fn=self.smoothing_fn,
                smoothing_temp=self.smoothing_temp
            )


        # Initialize accumulators on first call
        if self.granular_label_scores_sum is None:
#            self.total_scores_sum = torch.zeros(self.n_bins, dtype=self.dtype, device=self.device)
            self.granular_label_scores_sum = torch.zeros(self.n_labels, *self.extra_internal_label_dims_list, self.n_bins, dtype=self.dtype, device=self.device)
#            self.granular_label_counts = torch.zeros(self.num_labels, *self.extra_internal_label_dims_list, dtype=self.dtype, device=self.device)
            self.label_counts = torch.zeros(self.n_labels,dtype=self.dtype, device=self.device)



        ## ================================================
        ## ================================================


        ## 0. Create a tensor of the soft-binned probability distributions per granular label: 
        ## -----------------------------------------------------------------------------------
        ## Initialize the new tensor 
#        scores_by_label = torch.zeros(n_labels, *tuple(tmp_scores.shape[1:]), 
#                                    dtype=tmp_scores.dtype, device=tmp_scores.device)  
#                            ## [n_labels, *extra_internal_label_dims_list, n_bins]


        ## Create mask to only aggregate rows with for valid label indices
        valid_label_mask = index_tensor >= 0
        index_valid_flat = index_tensor[valid_label_mask]  ## [N_valid]
        scores_valid = scores[valid_label_mask]   ## [N_valid, *extra_internal_label_dims_list, n_bins]



        ## Accumulate the total number of rows processed
#        self.total_count += index_valid_flat.shape[0]

#        ## Accumulate the softbin distribution for all labels and all internal label dimensions
#        self.total_scores_sum += scores_valid.sum(dim = list(range(scores_valid.ndim - 1)))

        ## Sum the valid rows of tmp_scores into according to valid indices from index_tensor
        self.granular_label_scores_sum.index_add_(dim=0, source=scores_valid, index=index_valid_flat)

        ## Update the label counts
        self.label_counts += torch.bincount(index_valid_flat, minlength=self.n_labels)




        ## Expand index_tensor from shape [N_valid]
        ## to match the shape of tmp_scores: [N_valid, *extra_internal_label_dims_list, n_bins]
#        index_valid = index_valid_flat.view(-1, *([1] * (tmp_scores.ndim - 1)))     ## Add ones for the remaining number of dimensions
#        index_valid = index_valid.expand(-1, *(tmp_scores.shape[1:]))  ## Expand all remaining shapes to have the same values independent of these indices!

        #for _ in range(len(tmp_scores.shape[1:])):
        #   index_valid = index_valid.unsqueeze(-1)
        #index_valid.expand_as(tmp_scores_valid)



        _ = '''

        ## DIAGNOSTIC:
        print()
    #    print(f'n_labels = {n_labels}')
        print(f'index_tensor.shape = {index_tensor.shape}')
    #    print(f'index.shape = {index.shape}')
        print(f'index_valid.shape = {index_valid.shape}')
        print(f'type(tmp_scores) = {type(tmp_scores)}')
        print(f'tmp_scores.shape = {tmp_scores.shape}')
        print(f'tmp_scores_valid.shape = {tmp_scores_valid.shape}')
        print(f'type(scores_by_label) = {type(scores_by_label)}')
        print(f'scores_by_label.shape = {scores_by_label.shape}')
        print(f'scores_by_label.sum() = {scores_by_label.sum()}')
        print()


        ## Scatter-add: sum the valid rows of tmp_scores into according to indices from index_tensor
        scores_by_label.scatter_add_(0, index_valid, tmp_scores_valid) 


        ## DIAGNOSTIC:
        print()
        print("Now we've computed the scores_by_label tensor")
        print(f'scores_by_label.shape = {scores_by_label.shape}')
        print(f'scores_by_label.sum() = {scores_by_label.sum()}')
        print()

        '''


#        self.granular_label_scores_sum = None     # Shape: [num_labels, *extra_internal_label_dims_list, n_bins]
#        self.granular_label_counts = None         # Shape: [num_labels, *extra_internal_label_dims_list]


        ## ================================================
        ## ================================================

#        # Remove heads dimension
#        scores_no_heads = scores.squeeze(1)

#        # Initialize accumulators on first call
#        if self.total_scores_sum is None:
#            self.total_scores_sum = torch.zeros(self.n_bins, dtype=self.dtype, device=self.device)
#            self.label_scores_sum = torch.zeros(self.num_labels, self.n_bins, dtype=self.dtype, device=self.device)
#            self.label_counts = torch.zeros(self.num_labels, dtype=torch.long, device=self.device)


        ## Filter the unused rows


#        # Accumulate counts
#        self.total_count += scores.shape[0]
#        self.total_scores_sum += scores.sum(0)
#        self.granular_label_scores_sum.index_add_(dim=0, source=scores, index=index_tensor)
#        self.label_counts += torch.bincount(index_tensor, minlength=self.num_labels)



    ## Label-level entropies
    # entropy_with_contitions(given_specialized_variables=['label'], given_unspecialized_variables=['layer':[0]])

    ## Population-level entropies
    # entropy_with_contitionsgiven_unspecialized_variables=['layer'])


#    def entropy_with_conditions(self, given_specialized_variables_list=[], 
#                                      given_unspecialized_variables_list=[{'layer':[0,1,2,3,4,5,6]}, {'head':[0,1,2,3,4,5]}]):


    @torch.no_grad()
    def entropy_with_conditions(self, restrict_values_to_dict_of_value_lists={}, 
                                      given_variables_list=[],  
                                      SHOW_DIAGNOSTICS = False,
                                      ):
        """
        Computes a dictionary of entropies for all specified specialized variable values, 
        given the other specified unspecialized variables.

        Here the allowed entries of given_variables_list and keys of the dictionary
        restrict_values_to_dict_of_value_lists are given by self.label_name, the elements of
        self.extra_internal_label_dims_name_list and dimension_index_name ('dim_index' by default), 
        and the allowed values of restrict_values_to_dict_of_value_lists are lists of values 
        for the given key name that we would like to include as possibilities.  These value lists 
        can also be given (perhaps more appropriately) as sets.

        INPUTS:
            restrict_values_to_dict_of_value_lists = dict of lists or sets of allowed values for each key name.
            given_variables_list = list of (label name) strings

        """
        ## DIAGNOSTIC:
        if SHOW_DIAGNOSTICS:
            print(f'restrict_values_to_dict_of_value_lists = {restrict_values_to_dict_of_value_lists}')
            print(f'given_variables_list = {given_variables_list}')


        ## Alias the probability dimension label name
        dimension_index_name = self.probability_label_dim_name

        ## Get the number of indices in the granular label scores sum tensor
        shape_len = len(self.granular_label_scores_sum.shape)
        
        ## Make the list of labels for the granular soft-binning
        if dimension_index_name in [self.label_name] + self.label_list:
            raise RuntimeError(f"The name '{dimension_index_name}' already appears in the given label names!")
        granular_tensor_label_list = [self.label_name] + self.extra_internal_label_dims_name_list + [dimension_index_name]


        ## SANITY CHECK: Do the labels and the tensor have the same number of shape indices?
        if shape_len != len(granular_tensor_label_list):
            raise RuntimeError(f"shape_len = {shape_len} != len(granular_tensor_label_list) = {len(granular_tensor_label_list)}")

        
        ## Create index_selection_list -- 
        ## which translates our restructured values information into the desired format defining the sub-population:
        ##
        ##     index_selection_list = ['ALL', [1,3,5], 'ALL', 'ALL', 'ALL']
        ##
        ##
        index_selection_list = ['ALL'  for _ in range(shape_len)]
        for k, v_list in restrict_values_to_dict_of_value_lists.items():
            if k in granular_tensor_label_list:        
                k_index = granular_tensor_label_list.index(k)
                if k_index == 0:
                    v_list2 = [self.label_list.index(v)  for v in v_list]
                    index_selection_list[k_index] = list(set(v_list2))                
                else:
                    index_selection_list[k_index] = list(set(v_list))

        ## DIAGNOSTIC:
        if SHOW_DIAGNOSTICS:
            print(f"index_selection_list = {index_selection_list}")
            print()
        

        ## Create tmp_subpopulation_label_counts -- 
        ## which determines the appropriate label counts for the sub-population
        if index_selection_list[0] != 'ALL':
            tmp_subpopulation_label_counts = torch.tensor([self.label_counts[x]  for x in index_selection_list[0]])
        else:
            tmp_subpopulation_label_counts = sum(self.label_counts)

        
        ## Create known_dim_index_list -- 
        ## which determines the list of indices for the given known variables
        known_dim_index_list = []
        for var_name in given_variables_list:
            if var_name in granular_tensor_label_list:
                known_dim_index_list.append(granular_tensor_label_list.index(var_name))
            
        
        
        ## ===============================
        
        ## Filter to the sub-population specified by the value lists -- creating a copy for now:
        ## -------------------------------------------------------------------------------------

        ## Select the subset of indices we're interested in
        filtered_granular_scores_sum = self.granular_label_scores_sum
        for d, d_index_list in enumerate(index_selection_list):
            if d_index_list != 'ALL':
                #print(f'd = {d}')
                filtered_granular_scores_sum = filtered_granular_scores_sum.index_select(dim=d, index=torch.tensor(d_index_list))
        
        ## Compute the entropy separately for each value of the known dimensions we specify -- using the last index as the probability dsistribution
        #filtered_granular_entropies = entropy(filtered_granular_scores_sum)


        ## DIAGNOSTIC
        if SHOW_DIAGNOSTICS:
            print()
            print(f"self.granular_label_scores_sum.shape = {self.granular_label_scores_sum.shape}")
            print(f"filtered_granular_scores_sum.shape = {filtered_granular_scores_sum.shape}")
            print(f"d_index_list = {d_index_list}")
            print(f"self.label_list = {self.label_list}")
            print(f"self.label_counts = {self.label_counts}")
            print(f"tmp_subpopulation_label_counts = {tmp_subpopulation_label_counts}")
            print(f"")

        
        ## ===============================

        ## Combine these to get the probability distrbutions we're interested in    
        
        ## A1a. Sum over the unknown dimensions to get the probability sums for known populations by value -- first do non-label indices
        unknown_internal_dim_index_tuple = tuple([i  for i in range(shape_len-1)  if i not in known_dim_index_list and i > 0])
        if len(unknown_internal_dim_index_tuple) > 0:
            filtered_granular_scores_sum2 = filtered_granular_scores_sum.sum(dim=unknown_internal_dim_index_tuple, keepdim=True)
        else:
            filtered_granular_scores_sum2 = filtered_granular_scores_sum
        
        ## Get the number of internal summands
        number_of_internal_summands = torch.tensor([filtered_granular_scores_sum.shape[i]  \
                                                    for i in unknown_internal_dim_index_tuple  if i > 0]).prod()
        
        ## Divide to get the average of these probabilities over all internal populations
        filtered_granular_scores_sum2 = filtered_granular_scores_sum2 / number_of_internal_summands    


        ## DIAGNOSTIC
        if SHOW_DIAGNOSTICS:
            print()
            print("PART I -- Sum over the unknown dimensions to get the probability sums known populations:")
            print(f"known_dim_index_list = {known_dim_index_list}")
            print(f"unknown_internal_dim_index_tuple = {unknown_internal_dim_index_tuple}")
            print(f"number_of_internal_summands = {number_of_internal_summands}")
            #print(f"filtered_granular_scores_sum2 = {filtered_granular_scores_sum2}")
            print(f"filtered_granular_scores_sum2.shape = {filtered_granular_scores_sum2.shape}")
            print(f"")

        

        ## A1b. Now sum over the unknown label dimensions, each of which may have a different label multiplicity
        unknown_label_dim_index_tuple = tuple([i  for i in range(shape_len-1)  if i not in known_dim_index_list and i == 0])
        if len(unknown_label_dim_index_tuple) > 0:
            filtered_granular_scores_sum2 = filtered_granular_scores_sum2.sum(dim=unknown_label_dim_index_tuple, keepdim=True)
        else:
            filtered_granular_scores_sum2 = filtered_granular_scores_sum2
        
        ## Get the number of label summands we're including
        number_of_selected_label_summands = tmp_subpopulation_label_counts.sum().item()
        
        ## Divide to get the average of these probabilities over all selected label populations
        intermediate_probability_distributions = filtered_granular_scores_sum2 / number_of_selected_label_summands

        ## DIAGNOSTIC
        if SHOW_DIAGNOSTICS:
            print()
            print("PART II -- Sum over the unknown dimensions to get the probability sums known populations:")
            print(f"unknown_internal_dim_index_tuple = {unknown_internal_dim_index_tuple}")
            print(f"number_of_selected_label_summands = {number_of_selected_label_summands}")
            #print(f"intermediate_probability_distributions = {intermediate_probability_distributions}")
            print(f"intermediate_probability_distributions.shape = {intermediate_probability_distributions.shape}")
            print(f"")

        



        
        ## SANITY CHECK: Do each of these sum to 1?
        tmp_prob_tensor_sum = intermediate_probability_distributions.sum(dim=-1)
        #
        ## DIAGNOSTIC
        #print()
        #print(f"tmp_prob_tensor_sum.shape = {tmp_prob_tensor_sum.shape}")
        #print(f"tmp_prob_tensor_sum = {tmp_prob_tensor_sum}")
        #    
        if not torch.allclose(tmp_prob_tensor_sum, torch.ones_like(tmp_prob_tensor_sum)):
    #    if (tmp_sum_tensor == 1).all().item() == False:        
            raise RuntimeError("The intermediate probability distrubution sums are not all equal to 1!")




        
        ## B. Compute the entropies of these sums
        intermediate_entropies = entropy(intermediate_probability_distributions)
        
        
        ## C. Take the average value of the entropies we computed for each sub-population
        desired_entropy = intermediate_entropies.mean().item()


        ## Return the desired entropy
        return desired_entropy
    



    @torch.no_grad()
    def compute_metrics(self, conditional_entropy_label_weighting: Literal["weighted", "uniform"] = "weighted") -> dict:
        """
        Compute current entropy metrics from accumulated state.

        Args:
            conditional_entropy_label_weighting: Weighting scheme for conditional entropy
                - "weighted": Weight by empirical label probabilities
                - "uniform": Weight all labels equally

        Returns:
            Dictionary containing:
                - entropy: Total population entropy
                - conditional_entropy: Conditional entropy H(Z|L)
                - mutual_information: Mutual information I(Z;L)
                - label_entropy_dict: Per-label and total population entropies
                - intermediate_data: Probability distributions and bins
        """
        if self.total_count == 0:
            raise ValueError("Cannot compute metrics on empty accumulator. Call update() first.")

        # Normalize to get probability distributions
        prob_total = self.total_scores_sum / self.total_count
        prob_by_label = self.label_scores_sum / self.label_counts.unsqueeze(1)
        label_distribution = (self.label_counts / self.total_count).to(torch.float64)

        # Compute entropies
        total_entropy = entropy(prob_total)
        entropy_by_label = entropy(prob_by_label).to(torch.float64)

        # Build entropy dictionary
        entropy_dict = {'total_population': total_entropy.item()}
        for i, label in enumerate(self.label_list):
            entropy_dict[label] = entropy_by_label[i].item()

        # Conditional entropy
        if conditional_entropy_label_weighting == "weighted":
            cond_entropy = torch.dot(label_distribution, entropy_by_label).item()
        else:  # uniform
            n = len(self.label_list)
            uniform_dist = torch.ones(n, device=self.device) / n
            cond_entropy = torch.dot(uniform_dist, entropy_by_label).item()

        # Mutual information
        mutual_info = total_entropy.item() - cond_entropy

        return {
            'output_metrics': {
                'entropy': total_entropy.item(),
                'conditional_entropy': cond_entropy,
                'mutual_information': mutual_info,
                'label_entropy_dict': entropy_dict,
            },
            'intermediate_data': {
                'prob_dist_for_total_population_tensor': prob_total,
                'prob_dist_by_label_tensor': prob_by_label,
                'tmp_bins': self.bins,
            }
        }


    @torch.no_grad()
    def merge(self, other: 'EntropyAccumulator'):
        """
        Merge state from another accumulator for distributed computation.

        Args:
            other: Another EntropyAccumulator to merge into this one

        Raises:
            ValueError: If accumulators have incompatible configurations
        """
        # Validate compatibility
        if self.n_bins != other.n_bins:
            raise ValueError(f"Cannot merge: n_bins mismatch ({self.n_bins} != {other.n_bins})")
        if self.num_labels != other.num_labels:
            raise ValueError(f"Cannot merge: num_labels mismatch ({self.num_labels} != {other.num_labels})")
        if self.label_list != other.label_list:
            raise ValueError(f"Cannot merge: label_list mismatch")

        # If this accumulator is empty, adopt other's bins
        if self.bins is None:
            self.bins = other.bins
            self.dtype = other.dtype
            self.device = other.device
            self.total_scores_sum = other.total_scores_sum.clone() if other.total_scores_sum is not None else None
            self.label_scores_sum = other.label_scores_sum.clone() if other.label_scores_sum is not None else None
            self.label_counts = other.label_counts.clone() if other.label_counts is not None else None
            self.total_count = other.total_count
            return

        # If other is empty, nothing to merge
        if other.bins is None:
            return

        # Both have data - merge counts
        self.total_count += other.total_count
        self.total_scores_sum += other.total_scores_sum
        self.label_scores_sum += other.label_scores_sum
        self.label_counts += other.label_counts


    def get_state_dict(self) -> dict:
        """
        Get serializable state dictionary for saving/loading.

        Returns:
            Dictionary containing all accumulator state
        """
        return {
            'n_bins': self.n_bins,
            'label_list': self.label_list,
            'embedding_dim': self.embedding_dim,
            'n_heads': self.n_heads,
            'num_labels': self.num_labels,
            'dist_fn': self.dist_fn,
            'bin_type': self.bin_type,
            'smoothing_fn': self.smoothing_fn,
            'smoothing_temp': self.smoothing_temp,
            'total_count': self.total_count,
            'total_scores_sum': self.total_scores_sum,
            'label_scores_sum': self.label_scores_sum,
            'label_counts': self.label_counts,
            'bins': self.bins,
            'dtype': self.dtype,
            'device': str(self.device) if self.device is not None else None,
        }


    @classmethod
    def from_state_dict(cls, state_dict: dict) -> 'EntropyAccumulator':
        """
        Restore accumulator from state dictionary.

        Args:
            state_dict: Dictionary from get_state_dict()

        Returns:
            Restored EntropyAccumulator instance
        """
        acc = cls(
            n_bins=state_dict['n_bins'],
            label_list=state_dict['label_list'],
            embedding_dim=state_dict.get('embedding_dim'),  # Use .get() for backward compatibility
#            n_heads=state_dict['n_heads'],
            dist_fn=state_dict['dist_fn'],
            bin_type=state_dict['bin_type'],
            smoothing_fn=state_dict['smoothing_fn'],
            smoothing_temp=state_dict['smoothing_temp']
        )
        acc.num_labels = state_dict['num_labels']
        acc.total_count = state_dict['total_count']
        acc.total_scores_sum = state_dict['total_scores_sum']
        acc.label_scores_sum = state_dict['label_scores_sum']
        acc.label_counts = state_dict['label_counts']
        acc.bins = state_dict['bins']
        acc.dtype = state_dict['dtype']
        acc.device = torch.device(state_dict['device']) if state_dict['device'] is not None else None
        return acc


## ==========================================================================
## =================== BATCH ENTROPY COMPUTATION ============================
## ==========================================================================


## TO DO: REVISIT THIS TO COMPUTE THE CONDITIONAL ENTROPIES FOR EACH (SUB-)POPULATION!

@torch.no_grad()
def compute_all_entropy_measures2(
        data_embeddings_tensor: torch.Tensor,
        data_label_indices_tensor: torch.Tensor,
        label_list: list,
        n_bins: int = 10,
##        n_heads: int = 1,
        extra_internal_label_dims_list = [],
        extra_internal_label_dims_name_list = [],  ## e.g. ['layer output index plus-one (zero is pre-layers)', 'head index']
        dist_fn: str = 'euclidean',
        bin_type: str = 'uniform',
        smoothing_fn: str = 'None',
        smoothing_temp: float = 1.0,
        conditional_entropy_label_weighting: Literal["weighted", "uniform"] = "weighted",
        online_bins: Optional[torch.Tensor] = None, 
        SHOW_DIAGNOSTICS = False,
    ) -> dict:
    """
    Run the soft-binning and related entropy calculations on the given labelled emdeddings data.

    Args:
        data_embeddings_tensor: Data embeddings [N, *extra_internal_label_dims_list, D]
        data_label_indices_tensor: Label indices [N], values in range [0, num_labels)
        label_list: List of unique label names
        n_bins: Number of bins for soft-binning (default: 10)
##        n_heads: Number of attention heads (default: 1)
        dist_fn: Distance function for soft-binning (default: 'euclidean')
        bin_type: Binning strategy (default: 'uniform')
        smoothing_fn: Smoothing function (default: 'None')
        smoothing_temp: Temperature for smoothing (default: 1.0)
        conditional_entropy_label_weighting: "weighted" or "uniform" (default: "weighted")
        online_bins: Pre-computed bins to use instead of generating new ones (default: None)

    Returns:
        Dictionary containing entropy metrics and intermediate data

    Note:
        conditional_entropy_label_weighting is used in our computation of the
        conditional entropy, mutual_information, and the multi_JS_divergence.

        If online_bins is provided, it will be used instead of creating new bins. This is
        useful for ensuring consistency between batch and online computations.

    """
    ## Alias the inputs
    data_tensor = data_embeddings_tensor
    index_tensor = data_label_indices_tensor



    ## Perform the soft-binning
    tmp_scores, tmp_bins = \
        soft_bin2(all_representations = data_tensor, n_bins = n_bins, 
                 #n_heads = n_heads,
                 extra_internal_label_dims_list = extra_internal_label_dims_list,
                 #extra_internal_label_dims_name_list = extra_internal_label_dims_name_list,
                 dist_fn = dist_fn, bin_type = bin_type,
                 smoothing_fn = smoothing_fn, smoothing_temp = smoothing_temp,
                 online_bins = online_bins)
    ## tmp_scores shape = [N, *extra_internal_label_dims_list, n_bins]
    ## tmp_bins shape = [n_bins]

#    ## Get the data tensor with no extra n_heads variable -- REVISIT THIS!!
#    tmp_scores__no_heads = tmp_scores.squeeze(1)
#    tmp_scores__no_heads.shape

    ## Alias some useful sizes
    n_labels = len(label_list)


    ## DIAGNOSTIC:
    if SHOW_DIAGNOSTICS:
        print()
        print(f'n_labels = {n_labels}')
        print(f'index_tensor.shape = {index_tensor.shape}')
        print(f'type(tmp_scores) = {type(tmp_scores)}')
        print(f'tmp_scores.shape = {tmp_scores.shape}')
        print(f'tmp_scores.shape[1:] = {tmp_scores.shape[1:]}')
        #print(f'type(tmp_scores.shape[1:]) = {type(tmp_scores.shape[1:])}')
        #print(f'tuple(tmp_scores.shape[1:]) = {tuple(tmp_scores.shape[1:])}')
        #print(f'type(tuple(tmp_scores.shape[1:])) = {type(tuple(tmp_scores.shape[1:]))}')
        #print(f'tuple(tmp_scores.shape[1:])[0] = {tuple(tmp_scores.shape[1:])[0]}')
        #print(f'type(tuple(tmp_scores.shape[1:])[0]) = {type(tuple(tmp_scores.shape[1:])[0])}')
        print()


    ## 0. Create a tensor of the soft-binned probability distributions per granular label: 
    ## -----------------------------------------------------------------------------------
    ## Initialize the new tensor 
    scores_by_label = torch.zeros(n_labels, *tuple(tmp_scores.shape[1:]), 
                                  dtype=tmp_scores.dtype, device=tmp_scores.device)  
                        ## [n_labels, *extra_internal_label_dims_list, n_bins]


    ## Create mask to only aggregate rows with for valid label indices
    valid_label_mask = index_tensor >= 0
    index_valid_flat = index_tensor[valid_label_mask]  ## [N_valid]
    tmp_scores_valid = tmp_scores[valid_label_mask]   ## [N_valid, *extra_internal_label_dims_list, n_bins]


    ## Expand index_tensor from shape [N_valid]
    ## to match the shape of tmp_scores: [N_valid, *extra_internal_label_dims_list, n_bins]
    index_valid = index_valid_flat.view(-1, *([1] * (tmp_scores.ndim - 1)))     ## Add ones for the remaining number of dimensions
    index_valid = index_valid.expand(-1, *(tmp_scores.shape[1:]))  ## Expand all remaining shapes to have the same values independent of these indices!

    #for _ in range(len(tmp_scores.shape[1:])):
    #   index_valid = index_valid.unsqueeze(-1)
    #index_valid.expand_as(tmp_scores_valid)


    ## DIAGNOSTIC:
    if SHOW_DIAGNOSTICS:
        print()
    #    print(f'n_labels = {n_labels}')
        print(f'index_tensor.shape = {index_tensor.shape}')
    #    print(f'index.shape = {index.shape}')
        print(f'index_valid.shape = {index_valid.shape}')
        print(f'type(tmp_scores) = {type(tmp_scores)}')
        print(f'tmp_scores.shape = {tmp_scores.shape}')
        print(f'tmp_scores_valid.shape = {tmp_scores_valid.shape}')
        print(f'type(scores_by_label) = {type(scores_by_label)}')
        print(f'scores_by_label.shape = {scores_by_label.shape}')
        print(f'scores_by_label.sum() = {scores_by_label.sum()}')
        print()


    ## Scatter-add: sum the valid rows of tmp_scores into according to indices from index_tensor
    scores_by_label.scatter_add_(0, index_valid, tmp_scores_valid) 


    ## DIAGNOSTIC:
    if SHOW_DIAGNOSTICS:
        print()
        print("Now we've computed the scores_by_label tensor")
        print(f'scores_by_label.shape = {scores_by_label.shape}')
        print(f'scores_by_label.sum() = {scores_by_label.sum()}')
        print()


    ## SANITY CHECK:  THE FINAL DIMENSION SUMS SHOULD BE THE NUMBER OF DATA POINTS WITH THAT LABEL! =)


    _ = '''
    ## Return the desired quantities as a dictionary
    tmp_output_dict = {
        'diagnostic_data': {
            'data_tensor': data_tensor,
            'index_tensor': index_tensor,
            'label_list': label_list,
            'n_labels': n_labels,
            'tmp_scores': tmp_scores,
            'tmp_bins': tmp_bins,
            'valid_label_mask': valid_label_mask,
            'index_valid': index_valid,
            'tmp_scores_valid': tmp_scores_valid,
            'scores_by_label': scores_by_label,
        },
        'intermediate_data': {
#            'prob_dist_for_total_population_tensor': prob_dist_for_total_population_tensor,
#            'prob_dist_by_label_tensor': prob_dist_by_label_tensor,
            'tmp_bins': tmp_bins,
        },
    }
    return tmp_output_dict
    '''



    ## 1. Compute the probability distribution for the full dataset:
    ## -------------------------------------------------------------

    ## This is the probability distribution of the full dataset, here
    ##      scores_by_label --> [n_labels, *extra_internal_label_dims_list, n_bins]
    label_indices_tuple = tuple(range(scores_by_label.ndim - 1))
    population_size_with_internal_dims = torch.tensor(tmp_scores_valid.shape[:-1]).prod().item()  ## Accounts for the number of data points multiplied by all internal dimensions
    prob_dist_for_total_population_tensor = scores_by_label.sum(dim=label_indices_tuple) / population_size_with_internal_dims      ## shape = [n_bins]



    ## 2. Create the probability distributions for each population label:
    ## ------------------------------------------------------------------
    distribution_of_labels = (index_valid_flat.bincount() / index_valid_flat.shape[0]).to(torch.float64)

    ## We also want to know the distribution of labels -- this tensor should sum to N * prod(all internal dimensions).
    label_counts = torch.bincount(index_valid_flat, minlength=n_labels)
    internal_label_indices_tuple = tuple(range(1, scores_by_label.ndim - 1))

    #number_of_granular_summands = label_bin_size_tensor * (product of all internal dimension counts)
    number_of_internal_granular_summands = torch.tensor(scores_by_label.shape)[list(internal_label_indices_tuple)].prod().item()
    number_of_granular_summands_per_label = label_counts * number_of_internal_granular_summands

    ## Create the 1/(# of summands) tensor per label -- also accounting for when labels don't appear -- [n_labels]
    inverse_of_number_of_granular_summands_per_label = (1.0 / torch.where(number_of_granular_summands_per_label == 0, 1.0, number_of_granular_summands_per_label))

    ## Compute the probability distribution for each label
    prob_sum_by_label_tensor = scores_by_label.sum(dim=internal_label_indices_tuple)   ## [n_labels, n_bins]
    prob_dist_by_label_tensor = prob_sum_by_label_tensor * inverse_of_number_of_granular_summands_per_label[:, None]





    _ = '''
    
    ## Return the desired quantities as a dictionary
    tmp_output_dict = {
        'diagnostic_data': {
            'data_tensor': data_tensor,
            'index_tensor': index_tensor,
            'label_list': label_list,
            'n_labels': n_labels,
            'tmp_scores': tmp_scores,
            'tmp_bins': tmp_bins,
            'valid_label_mask': valid_label_mask,
            'index_valid': index_valid,
            'tmp_scores_valid': tmp_scores_valid,
            'scores_by_label': scores_by_label,
        },
        'intermediate_data': {
            'prob_dist_for_total_population_tensor': prob_dist_for_total_population_tensor,
            'prob_dist_by_label_tensor': prob_dist_by_label_tensor,
            'tmp_bins': tmp_bins,
        },
    }
    return tmp_output_dict

    '''






    ## GENERAL ROUTINE:
    ## ----------------

    ## To compute a probability distribution for a given population (e.g. all labels in layer 2)
    ##   - start with the fully granular label - soft-bin sum tensor, and the label counts (which trickle-down to the granular counts)
    ##   - determine the sub-population to sum over
    ##   - divide the sum by the number of summands to get the subpopulation entropy

    ## Example:
    ##   - filter for the given values -- e.g. layer 2
    ##   - compute the sum of all sum
    ## 
    ## 

    ## SYNTAX: 
    ##   entropy({'label':['red', 'blue'] , 'layer':[3,4] , 'head':[1]})
    ##   --> {'entropy': 2.034,  'cumulative_softbin_tensor': tensor([0.34, 0.12, 0.66]), 
    ##        'population_size':60, 'population_granular_softbin_tensor': tensor(...)}

    ## NOTE: We need to check that we are dealing with unpopulated labels correctly -- we replace the 1/N normalizing factor with 1 or zero.





    ## 4. Compute the entropy and related metrics:
    ## -------------------------------------------

    ## Compute the entropy for the total population
    total_population_entropy = entropy(prob_dist_for_total_population_tensor)
    #total_population_entropy

    ## Compute the entropy for each label (sub-population)
    entropy_by_label_tensor = entropy(prob_dist_by_label_tensor).to(dtype=torch.float64)
    #entropy_by_label_tensor

    ## Store the (sub-)population entropies in a dictionary for easy reference
    entropy_dict = {
        'total_population': total_population_entropy.item()
    }
    for i, label in enumerate(label_list):
        entropy_dict[label] = entropy_by_label_tensor[i].item()


    ## Compute the conditional entropy for the population given the categorical label -- to options for this!
    if conditional_entropy_label_weighting == "weighted":
        conditional_entropy_of_population_given_the_label = torch.dot(distribution_of_labels, entropy_by_label_tensor).item()
    elif conditional_entropy_label_weighting == "uniform":
        n = distribution_of_labels.shape[0]
        uniform_dist_of_labels = torch.ones(n) / (1.0 * n)
        conditional_entropy_of_population_given_the_label = torch.dot(uniform_dist_of_labels, entropy_by_label_tensor).item()
        
    ## Compute the mutual information given the conditional entropy
    mutual_information = entropy_dict['total_population'] - conditional_entropy_of_population_given_the_label

    ## Compute the multi-JS Divergence
    #multi_JS_div = multi_js_divergence(classes=tmp_scores__no_heads, p_class=distribution_of_labels, 
    #                    max_normalization=conditional_entropy_label_weighting)
    



    ## Return the desired quantities as a dictionary
    output_dict = {
        'diagnostic_data': {
            'data_tensor': data_tensor,
            'index_tensor': index_tensor,
            'label_list': label_list,
            'n_labels': n_labels,
            'tmp_scores': tmp_scores,
            'tmp_bins': tmp_bins,
            'valid_label_mask': valid_label_mask,
            'index_valid': index_valid,
            'tmp_scores_valid': tmp_scores_valid,
            'scores_by_label': scores_by_label,
        },
        'intermediate_data': {
            'prob_dist_for_total_population_tensor': prob_dist_for_total_population_tensor,
            'prob_dist_by_label_tensor': prob_dist_by_label_tensor,
            'tmp_bins': tmp_bins,
        },
        'output_metrics': {
            'entropy': entropy_dict['total_population'],
            'conditional_entropy': conditional_entropy_of_population_given_the_label,
            'mutual_information': mutual_information,
            #'multi-JS_divergence': multi_JS_div,
            'label_entropy_dict': entropy_dict,
            'extra_internal_label_dims_list' : extra_internal_label_dims_list,
            'extra_internal_label_dims_name_list' : extra_internal_label_dims_name_list,
        }    
    }
    return output_dict




