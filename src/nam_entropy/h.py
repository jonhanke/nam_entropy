"""
Soft Entropy Calculation Module (new_h.py)
==========================================

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
#from sklearn.cluster import KMeans




## ==========================================================================
## =================== SOFT-BINNING CALCULATION ROUTINES ====================
## ==========================================================================


def soft_bin(all_representations, n_bins, bins=None, centers=None, 
            temp=1.0, dist_fn='cosine', bin_type='uniform', sub_mean=False, n_heads=4,
            smoothing_fn="softmax", online_bins=None, set_var=1.0, online_var=None, 
            show_diagnostics=False):        
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
        tuple: (scores, centres, centres, bins, online_var)
            - scores: Soft assignment probabilities [N, n_heads, n_bins]
            - centres: Bin centers
            - bins: Bin locations used for scoring
            - online_var: Updated variance statistics

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
    scores = distance(all_representations, bins, dist_fn)  ## Returns [N, n_heads, n_bins]
    scores = smoothing(scores, temp, smoothing_fn)   ## Takes / Returns: [N, n_heads, n_bins]

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
    - 'N_unit_sphere': Scaled unit sphere bins (N=10,50,100)
    - 'unit_cube': Evenly spaced bins within data range
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
        bins = F.normalize_by_scaling(bins, dim=-1)
       
    elif bin_type == 'unit_cube':
        bins = unit_cube_bins(
            start=all_representations.min(0).values,
            stop=all_representations.max(0).values,
            n_bins=n_bins
        ).T
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
        
    elif bin_type == "cluster":
        bins  = cluster(
            all_representations, n_bins,
        )
        bins = bins.view(n_bins, n_heads, int(d_hidden/n_heads))
    else:
        raise NotImplementedError
        
    bins = bins.to(all_representations.device)
    return bins
    



def distance(all_representations: torch.Tensor, 
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
        all_representations = F.normalize_by_scaling(all_representations, dim=-1)*5
        bins = F.normalize_by_scaling(bins, dim=-1)*5
        
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



def cluster(all_representations, n_bins, just_bins=True):
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
## ==================== MEASUREMENT REPORTING ROUTINES ======================
## ==========================================================================


def multi_js_divergence(classes: torch.Tensor, p_class: torch.Tensor, 
                        max_normalisation: str = "weighted") -> torch.Tensor:
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


    DEVELOPER NOTES:
        - The classes are tensors whose last index runs over the finite set 
            of points where the probability is supported.


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


    ## Normalisation bounds (theoretical maximum JS divergence -- uniform and weighted)
    uniform_entropy = torch.log(torch.tensor(p_class.shape[0]))   ## Max at the uniform distribution
    weighted_class_entropy = entropy(p_expanded.T, normalisation=None).mean()   ## p_expanded.T ==> [1, n_classes]
    
    ## Compute the desired normalized JS divergence
    allowed_max_normalisations = ["uniform", "weighted"]
    if max_normalisation == "uniform":
        result = js_divs / uniform_entropy
    elif max_normalisation == "weighted":
        result = js_divs / weighted_class_entropy
    else:
        raise ValueError(f"max_normalisation = {max_normalisation} must be in {allowed_max_normalisations}.")
    
    ## Return the desired result
    return result





def js_divergence(p: torch.Tensor, q: torch.Tensor, 
                  eps: float = 1e-9, use_xlogy: bool = True,
                  normalisation: str = None) -> torch.Tensor:
    """
    Computes Jensen-Shannon divergence between two distributions.
    
    The JS divergence is symmetric and bounded between 0 and ln(2).
    It measures the similarity between two probability distributions.
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Args:
        p, q (torch.Tensor): Input distributions. Last dimension should contain
                           the probability values.
        normalisation (str, optional): How to normalize inputs if the last index values don't sum to 1:
                                     - None: No normalization (assume already normalized)
                                     - "scaling": Divide by sum (L1 normalisation)  
                                     - "softmax": Apply softmax normalisation
        eps (float): Small epsilon for numerical stability. Default: 1e-9

    Returns:
        torch.Tensor: JS divergence score(s)
        
    Raises:
        ValueError: If the normalisation parameter is invalid
        
    Note:
        - The input tensors must have the same shape or be broadcastable
        - Normalisation is applied along the last dimension
        - For numerical stability when use_xlogy is False, a small epsilon 
            (clamping) parameter is present to address issues at log(0).
    """
    ## Validate the normalisation parameter
    valid_normalisations = {None, "scaling", "softmax"}
    if normalisation not in valid_normalisations:
        raise ValueError(f"Invalid normalization '{normalisation}'. "
                        f"Must be one of {valid_normalisations}")

    ## Normalize the distribution in case it doesn't already sum to 1
    if normalisation is not None:
        if normalisation == "scaling":
            p = normalize_by_scaling(p)
            q = normalize_by_scaling(q)
        elif normalisation == "softmax":
            p = normalize_by_softmax(p)
            q = normalize_by_softmax(q)

    ## Compute mean / mixture distribution  m = 0.5 * (p + q)
    m = 0.5 * (p + q)

    ## Compute the average of the two KL-divergences with m
    js_div = 0.5 * kl_divergence(p, m, normalisation=None, eps=eps, use_xlogy=use_xlogy) + \
             0.5 * kl_divergence(q, m, normalisation=None, eps=eps, use_xlogy=use_xlogy)
    
    ## Return the desired value
    return js_div




def kl_divergence(p: torch.Tensor, q: torch.Tensor, 
                  eps: float = 1e-9, use_xlogy: bool = True, 
                  normalisation: str = None) -> torch.Tensor:
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
    ## Validate the normalisation parameter
    valid_normalisations = {None, "scaling", "softmax"}
    if normalisation not in valid_normalisations:
        raise ValueError(f"Invalid normalization '{normalisation}'. "
                        f"Must be one of {valid_normalisations}")

    ## Normalize the distribution in case it doesn't already sum to 1
    if normalisation is not None:
        if normalisation == "scaling":
            p = normalize_by_scaling(p)
            q = normalize_by_scaling(q)
        elif normalisation == "softmax":
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
    ## Validate the normalisation parameter
    valid_normalizations = {None, "scaling", "softmax"}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. "
                        f"Must be one of {valid_normalisztions}")

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

    



def compute_all_entropy_measures(data_embeddings_tensor, data_label_indices_tensor, label_list, 
                                 n_bins=10, n_heads=1, 
                                 conditional_entropy_label_weighting="weighted"):
    """
    Run the soft-binning and related entropy calculations on the given labelled emdeddings data.

    Here conditional_entropy_label_weighting = "weighted" or "uniform".

    This is used in our computation of the conditional entropy, mutual_information, and the multi_JS_divergence.

    """
    ## Alias the inputs
    data_tensor = data_embeddings_tensor
    index_tensor = data_label_indices_tensor


    
    ## Perform the soft-binning
    tmp_scores, tmp_bins = \
        soft_bin(all_representations = data_tensor, n_bins = n_bins, n_heads = n_heads)

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
    #                    max_normalisation=conditional_entropy_label_weighting)

    
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
        }    
    }
    return output_dict
    
    