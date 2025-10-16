"""
Soft Entropy Calculation Module (h.py)
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



## ==========================================================================
## =================== SOFT-BINNING CALCULATION ROUTINES ====================
## ==========================================================================



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
                 n_heads: int = 1, dist_fn: str = 'euclidean', bin_type: str = 'uniform',
                 smoothing_fn: str = 'None', smoothing_temp: float = 1.0):
        """
        Initialize the entropy accumulator.

        Args:
            n_bins: Number of bins for soft-binning
            label_list: List of unique label names
            embedding_dim: Dimension of data embeddings. Required for data-independent bin types
                          like 'unit_sphere', 'standard_normal'. Optional for data-dependent
                          bin types like 'uniform'. If provided, bins are pre-computed.
            n_heads: Number of attention heads (default: 1)
            dist_fn: Distance function for soft-binning (default: 'euclidean')
            bin_type: Binning strategy (default: 'uniform')
            smoothing_fn: Smoothing function (default: 'None')
            smoothing_temp: Temperature for smoothing (default: 1.0)
        """
        self.n_bins = n_bins
        self.label_list = label_list
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
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
## =================== BATCH ENTROPY COMPUTATION ============================
## ==========================================================================


def compute_all_entropy_measures(
        data_embeddings_tensor: torch.Tensor,
        data_label_indices_tensor: torch.Tensor,
        label_list: list,
        n_bins: int = 10,
        n_heads: int = 1,
        dist_fn: str = 'euclidean',
        bin_type: str = 'uniform',
        smoothing_fn: str = 'None',
        smoothing_temp: float = 1.0,
        conditional_entropy_label_weighting: Literal["weighted", "uniform"] = "weighted",
        online_bins: Optional[torch.Tensor] = None
    ) -> dict:
    """
    Run the soft-binning and related entropy calculations on the given labelled emdeddings data.

    Args:
        data_embeddings_tensor: Data embeddings [N, D]
        data_label_indices_tensor: Label indices [N], values in range [0, num_labels)
        label_list: List of unique label names
        n_bins: Number of bins for soft-binning (default: 10)
        n_heads: Number of attention heads (default: 1)
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
        soft_bin(all_representations = data_tensor, n_bins = n_bins, n_heads = n_heads,
                 dist_fn = dist_fn, bin_type = bin_type,
                 smoothing_fn = smoothing_fn, smoothing_temp = smoothing_temp,
                 online_bins = online_bins)

    ## Get the data tensor with no extra n_heads variable
    tmp_scores__no_heads = tmp_scores.squeeze(1)
    tmp_scores__no_heads.shape



    ## 1. Compute the probability distribution for the full dataset:
    ## -------------------------------------------------------------
    
    ## Compute the sum of all soft-binned probability distibutions
    prob_dist_sum_tensor = tmp_scores__no_heads.sum(0)
    
    ## Compute the total population probability vector
    prob_dist_for_total_population_tensor = prob_dist_sum_tensor / tmp_scores__no_heads.shape[0]



    ## 2. Create the probability distributions for each population label:
    ## ------------------------------------------------------------------
    
    ## Prepare to compute the index sum
    num_samples = index_tensor.shape[0]  ## also data_tensor.shape[0]
    n_bins = tmp_scores.shape[-1]  ## prob_dist_num_of_points
    num_labels = len(label_list)
    
    ## Get the data tensor with no extra n_heads variable
    tmp_scores__no_heads = tmp_scores.squeeze(1)
    
    ## Compute the sum of the soft-binned probability distributions for each label
    label_prob_dist_sum_tensor = torch.zeros(num_labels, n_bins, dtype = tmp_scores__no_heads.dtype)
    label_prob_dist_sum_tensor = label_prob_dist_sum_tensor.index_add(dim=0, source=tmp_scores__no_heads, index=index_tensor)


    ## Determine the label counts (i.e. the number of samples for each label)
    label_counts_tensor = torch.bincount(index_tensor)
    
    ## Divide by the label counts to get the probability distributions of each label as a row
    label_prob_dist_avg_tensor = label_prob_dist_sum_tensor / label_counts_tensor.unsqueeze(1)

    ## Define the probability distributions for each label
    prob_dist_by_label_tensor = label_prob_dist_avg_tensor


    
    ## 3. Compute the probabilities of the labels:
    ## -------------------------------------------
    distribution_of_labels = (index_tensor.bincount() / index_tensor.shape[0]).to(torch.float64)
    #distribution_of_labels

    

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
        }    
    }
    return output_dict
    
    