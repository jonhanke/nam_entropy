"""
Utility functions for the NAM Entropy package.

This module provides helper functions for common tasks like setting random seeds
for reproducibility across different libraries.
"""

import random
import numpy as np
import torch


def set_all_random_seeds(seed: int, verbose: bool = True, make_deterministic: bool = True) -> None:
    """
    Set random seeds for all common libraries to ensure reproducibility.

    This function sets the random seed for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - SciPy (via NumPy's random state)

    Args:
        seed: The random seed value to use across all libraries
        verbose: If True, print confirmation message (default: True)
        make_deterministic: If True, set PyTorch to deterministic mode (default: True)
                           Note: This may impact performance but ensures full reproducibility

    Example:
        >>> from nam_entropy.utils import set_all_random_seeds
        >>> set_all_random_seeds(42)
        All random seeds set to 42

        >>> # To suppress the confirmation message
        >>> set_all_random_seeds(42, verbose=False)

        >>> # To allow non-deterministic behavior (faster but less reproducible)
        >>> set_all_random_seeds(42, make_deterministic=False)

    Note:
        - SciPy uses NumPy's random state, so setting np.random.seed() handles it
        - Setting make_deterministic=True may reduce performance in PyTorch operations
        - For distributed training, you may need additional seed setting per worker
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy (also handles SciPy)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch deterministic (optional, may impact performance)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose:
        deterministic_str = " (deterministic mode)" if make_deterministic else ""
        print(f"All random seeds set to {seed}{deterministic_str}")
