"""
Validation utilities for entropy accumulator and related data structures.

This module provides comprehensive validation functions to verify invariants
and catch bugs in the entropy accumulation pipeline.
"""

import torch
from typing import Optional, List, Tuple, Dict, Any


def validate_numerical_integrity(
    accumulator,
    check_nan: bool = True,
    check_inf: bool = True,
    check_non_negative: bool = True,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates numerical integrity of accumulator values.

    Checks for NaN/Inf values and non-negativity, which can arise from
    numerical instability (softmax overflow, division by zero, etc.).

    Args:
        accumulator: EntropyAccumulator2 instance to validate
        check_nan: Check for NaN values. Default: True
        check_inf: Check for Inf values. Default: True
        check_non_negative: Check that all values are >= 0. Default: True
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])

    Raises:
        AssertionError: If validation fails and raise_on_failure=True
        RuntimeError: If accumulators not initialized
    """
    if accumulator.granular_label_scores_sum is None:
        raise RuntimeError("Accumulators not initialized. Call update() with data first.")

    errors = []

    # Check granular_label_scores_sum
    scores = accumulator.granular_label_scores_sum
    if check_nan and torch.isnan(scores).any():
        nan_count = torch.isnan(scores).sum().item()
        errors.append(f"granular_label_scores_sum contains {nan_count} NaN values")

    if check_inf and torch.isinf(scores).any():
        inf_count = torch.isinf(scores).sum().item()
        errors.append(f"granular_label_scores_sum contains {inf_count} Inf values")

    if check_non_negative and (scores < 0).any():
        neg_count = (scores < 0).sum().item()
        min_val = scores.min().item()
        errors.append(f"granular_label_scores_sum contains {neg_count} negative values (min={min_val})")

    # Check label_counts
    counts = accumulator.label_counts
    if check_nan and torch.isnan(counts).any():
        errors.append("label_counts contains NaN values")

    if check_inf and torch.isinf(counts).any():
        errors.append("label_counts contains Inf values")

    if check_non_negative and (counts < 0).any():
        neg_labels = torch.where(counts < 0)[0].tolist()
        errors.append(f"label_counts has negative values at indices: {neg_labels}")

    # Check bins if present
    if accumulator.bins is not None:
        bins = accumulator.bins
        if check_nan and torch.isnan(bins).any():
            errors.append("bins contains NaN values")
        if check_inf and torch.isinf(bins).any():
            errors.append("bins contains Inf values")

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Numerical integrity validation failed:\n" + "\n".join(errors))

    return passed, errors


def validate_shape_consistency(
    accumulator,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that all accumulator shapes are internally consistent.

    Checks:
        - len(label_list) == n_labels == granular_label_scores_sum.shape[0] == label_counts.shape[0]
        - n_bins == granular_label_scores_sum.shape[-1]
        - bins.shape[0] == n_bins (if bins initialized)
        - extra_internal_label_dims_list matches middle dimensions of granular_label_scores_sum

    Args:
        accumulator: EntropyAccumulator2 instance to validate
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])

    Raises:
        AssertionError: If validation fails and raise_on_failure=True
    """
    errors = []

    # Check label dimensions
    n_labels = accumulator.n_labels
    label_list_len = len(accumulator.label_list)

    if label_list_len != n_labels:
        errors.append(f"len(label_list)={label_list_len} != n_labels={n_labels}")

    if accumulator.granular_label_scores_sum is not None:
        scores_shape = accumulator.granular_label_scores_sum.shape

        if scores_shape[0] != n_labels:
            errors.append(f"granular_label_scores_sum.shape[0]={scores_shape[0]} != n_labels={n_labels}")

        if accumulator.label_counts is not None:
            counts_shape = accumulator.label_counts.shape
            if counts_shape[0] != n_labels:
                errors.append(f"label_counts.shape[0]={counts_shape[0]} != n_labels={n_labels}")

        # Check n_bins
        n_bins = accumulator.n_bins
        if scores_shape[-1] != n_bins:
            errors.append(f"granular_label_scores_sum.shape[-1]={scores_shape[-1]} != n_bins={n_bins}")

        # Check bins shape
        if accumulator.bins is not None:
            bins_shape = accumulator.bins.shape
            if bins_shape[0] != n_bins:
                errors.append(f"bins.shape[0]={bins_shape[0]} != n_bins={n_bins}")

        # Check extra_internal_label_dims_list matches middle dimensions
        expected_middle_dims = tuple(accumulator.extra_internal_label_dims_list)
        actual_middle_dims = scores_shape[1:-1]
        if actual_middle_dims != expected_middle_dims:
            errors.append(
                f"granular_label_scores_sum middle dims {actual_middle_dims} != "
                f"extra_internal_label_dims_list {expected_middle_dims}"
            )

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Shape consistency validation failed:\n" + "\n".join(errors))

    return passed, errors


def validate_label_counts_are_integers(
    accumulator,
    rtol: float = 1e-9,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that label_counts are (close to) exact integers.

    Since label_counts represents the number of samples per label, it should
    be an exact integer. With float64 accumulators, this should hold precisely.

    Args:
        accumulator: EntropyAccumulator2 instance to validate
        rtol: Relative tolerance for integer check. Default: 1e-9
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])
    """
    if accumulator.label_counts is None:
        raise RuntimeError("Accumulators not initialized. Call update() with data first.")

    errors = []
    counts = accumulator.label_counts

    # Check if counts are close to integers
    rounded = torch.round(counts)
    diff = torch.abs(counts - rounded)
    max_diff = diff.max().item()

    if max_diff > rtol:
        non_integer_mask = diff > rtol
        non_integer_indices = torch.where(non_integer_mask)[0].tolist()
        non_integer_values = counts[non_integer_mask].tolist()
        errors.append(
            f"label_counts has non-integer values at indices {non_integer_indices}: "
            f"{non_integer_values} (max deviation from integer: {max_diff})"
        )

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Label counts integer validation failed:\n" + "\n".join(errors))

    return passed, errors


def validate_probability_distributions(
    accumulator,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that normalized score distributions are valid probability distributions.

    After normalizing granular_label_scores_sum by label_counts, each distribution
    should have values in [0, 1] and sum to 1.

    Args:
        accumulator: EntropyAccumulator2 instance to validate
        rtol: Relative tolerance for sum-to-1 check. Default: 1e-5
        atol: Absolute tolerance for sum-to-1 check. Default: 1e-8
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])
    """
    if accumulator.granular_label_scores_sum is None:
        raise RuntimeError("Accumulators not initialized. Call update() with data first.")

    errors = []
    scores = accumulator.granular_label_scores_sum
    counts = accumulator.label_counts

    # For each label with non-zero count, normalize and check
    for label_idx in range(scores.shape[0]):
        count = counts[label_idx].item()
        if count == 0:
            continue

        label_name = accumulator.label_list[label_idx] if label_idx < len(accumulator.label_list) else f"index_{label_idx}"

        # Get scores for this label: [*extra_internal_label_dims_list, n_bins]
        label_scores = scores[label_idx]

        # Normalize by count to get probability distribution
        # Sum over bins should give count, so dividing by count should give sum=1
        normalized = label_scores / count

        # Check values in [0, 1]
        if (normalized < -atol).any():
            min_val = normalized.min().item()
            errors.append(f"Label '{label_name}': normalized distribution has negative values (min={min_val})")

        if (normalized > 1 + atol).any():
            max_val = normalized.max().item()
            errors.append(f"Label '{label_name}': normalized distribution has values > 1 (max={max_val})")

        # Check sums to 1 (for each internal dimension combination)
        sums = normalized.sum(dim=-1)  # Sum over bins
        expected = torch.ones_like(sums)
        if not torch.allclose(sums, expected, rtol=rtol, atol=atol):
            min_sum = sums.min().item()
            max_sum = sums.max().item()
            errors.append(
                f"Label '{label_name}': normalized distribution sums range [{min_sum}, {max_sum}], expected 1.0"
            )

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Probability distribution validation failed:\n" + "\n".join(errors))

    return passed, errors


def validate_entropy_bounds(
    entropy_dict: Dict[Any, float],
    n_bins: int,
    check_non_negative: bool = True,
    check_upper_bound: bool = True,
    atol: float = 1e-6,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that computed entropies are within valid bounds.

    Entropy should be:
        - Non-negative (H >= 0)
        - Bounded above by log(n_bins) for discrete distributions

    Args:
        entropy_dict: Dictionary mapping labels/conditions to entropy values (float or tensor)
        n_bins: Number of bins (determines upper bound)
        check_non_negative: Check H >= 0. Default: True
        check_upper_bound: Check H <= log(n_bins). Default: True
        atol: Absolute tolerance for bound checks. Default: 1e-6
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])
    """
    import math

    errors = []
    max_entropy = math.log(n_bins)

    for key, entropy in entropy_dict.items():
        # Convert tensor to float if needed
        entropy_val = entropy.item() if hasattr(entropy, 'item') else float(entropy)

        if check_non_negative and entropy_val < -atol:
            errors.append(f"Entropy for {key} is negative: {entropy_val}")

        if check_upper_bound and entropy_val > max_entropy + atol:
            errors.append(
                f"Entropy for {key} exceeds maximum: {entropy_val} > log({n_bins}) = {max_entropy}"
            )

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Entropy bounds validation failed:\n" + "\n".join(errors))

    return passed, errors


def validate_information_inequalities(
    conditional_entropy: float,
    unconditional_entropy: float,
    mutual_information: Optional[float] = None,
    atol: float = 1e-6,
    raise_on_failure: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates information-theoretic inequalities.

    Checks:
        - H(X|Y) <= H(X): Conditioning reduces entropy
        - I(X;Y) >= 0: Mutual information is non-negative
        - I(X;Y) = H(X) - H(X|Y): Mutual information identity (if MI provided)

    Args:
        conditional_entropy: H(X|Y) - float or tensor
        unconditional_entropy: H(X) - float or tensor
        mutual_information: I(X;Y), optional - float or tensor
        atol: Absolute tolerance. Default: 1e-6
        raise_on_failure: Raise AssertionError on failure. Default: True

    Returns:
        Tuple of (passed: bool, error_messages: List[str])
    """
    # Convert tensors to floats if needed
    def to_float(x):
        return x.item() if hasattr(x, 'item') else float(x)

    cond_h = to_float(conditional_entropy)
    uncond_h = to_float(unconditional_entropy)
    mi = to_float(mutual_information) if mutual_information is not None else None

    errors = []

    # H(X|Y) <= H(X)
    if cond_h > uncond_h + atol:
        errors.append(
            f"Conditional entropy exceeds unconditional: H(X|Y)={cond_h} > H(X)={uncond_h}"
        )

    if mi is not None:
        # I(X;Y) >= 0
        if mi < -atol:
            errors.append(f"Mutual information is negative: I(X;Y)={mi}")

        # I(X;Y) = H(X) - H(X|Y)
        expected_mi = uncond_h - cond_h
        if abs(mi - expected_mi) > atol:
            errors.append(
                f"Mutual information inconsistent: I(X;Y)={mi} != "
                f"H(X) - H(X|Y) = {uncond_h} - {cond_h} = {expected_mi}"
            )

    passed = len(errors) == 0

    if not passed and raise_on_failure:
        raise AssertionError("Information inequality validation failed:\n" + "\n".join(errors))

    return passed, errors


def _wrap_accumulator_sums_validation(accumulator, rtol: float, atol: float) -> Tuple[bool, List[str]]:
    """
    Wrapper for accumulator.validate_accumulator_sums to capture error messages.

    The built-in method returns only a bool when raise_on_failure=False,
    losing error details. This wrapper captures errors by catching the exception.
    """
    try:
        # Run with raise_on_failure=True to get detailed error message
        accumulator.validate_accumulator_sums(rtol=rtol, atol=atol, raise_on_failure=True)
        return True, []
    except AssertionError as e:
        # Extract error message from exception
        return False, [str(e)]
    except RuntimeError as e:
        # Handle uninitialized accumulator
        return False, [str(e)]


def validate_all(
    accumulator,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    raise_on_failure: bool = True,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Tuple[bool, List[str]]]]:
    """
    Runs all accumulator validations and reports results.

    Validations performed:
        1. Numerical integrity (NaN, Inf, non-negativity)
        2. Shape consistency
        3. Label counts are integers
        4. Accumulator sums match label counts (calls accumulator.validate_accumulator_sums)
        5. Probability distributions are valid

    Args:
        accumulator: EntropyAccumulator2 instance to validate
        rtol: Relative tolerance for comparisons. Default: 1e-5
        atol: Absolute tolerance for comparisons. Default: 1e-8
        raise_on_failure: Raise AssertionError on first failure. Default: True
        verbose: Print status for each validation. Default: False

    Returns:
        Tuple of (all_passed: bool, results: Dict mapping validation name to (passed, errors))

    Raises:
        AssertionError: If any validation fails and raise_on_failure=True
    """
    results = {}
    all_passed = True

    validations = [
        ("numerical_integrity", lambda: validate_numerical_integrity(accumulator, raise_on_failure=False)),
        ("shape_consistency", lambda: validate_shape_consistency(accumulator, raise_on_failure=False)),
        ("label_counts_integers", lambda: validate_label_counts_are_integers(accumulator, raise_on_failure=False)),
        ("accumulator_sums", lambda: _wrap_accumulator_sums_validation(accumulator, rtol, atol)),
        ("probability_distributions", lambda: validate_probability_distributions(accumulator, rtol=rtol, atol=atol, raise_on_failure=False)),
    ]

    for name, validate_fn in validations:
        try:
            passed, errors = validate_fn()
        except Exception as e:
            passed = False
            errors = [str(e)]

        results[name] = (passed, errors)

        if verbose:
            status = "PASSED" if passed else "FAILED"
            print(f"  {name}: {status}")
            if not passed:
                for err in errors:
                    print(f"    - {err}")

        if not passed:
            all_passed = False
            if raise_on_failure:
                raise AssertionError(
                    f"Validation '{name}' failed:\n" + "\n".join(errors)
                )

    return all_passed, results
