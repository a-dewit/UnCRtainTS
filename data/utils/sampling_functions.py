from typing import List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for better readability
IndexList = List[int]
CoverageArray = NDArray[np.float32]
SamplingMethod = Literal["random", "fixedsubset", "fixed"]


def fixed_sampler(n_input_t: int, min_cov: float, max_cov: float, coverage: CoverageArray, clear_tresh: float = 1e-3) -> Tuple[IndexList, int, bool]:
    """
    Sample time points with fixed cloud coverage criteria.

    Selects the least cloudy sample as target and tries to find input samples within
    specified cloud coverage range. Falls back to first n samples if criteria not met.

    Args:
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold for input samples
        max_cov: Maximum cloud coverage threshold for input samples
        coverage: Array of cloud coverage values for each time point
        clear_tresh: Maximum coverage threshold for target sample

    Returns:
        Tuple containing:
        - List of input sample indices
        - Index of clearest (target) sample
        - Boolean indicating if coverage criteria were met
    """
    # Sort observation indices by cloud coverage (ascending)
    coverage_idx = np.argsort(coverage)
    cloudless_idx = coverage_idx[0]  # Take the (earliest) least cloudy sample

    # Select input samples within specified coverage range
    inputs_idx = [pdx for pdx, perc in enumerate(coverage) if min_cov <= perc <= max_cov][:n_input_t]

    # Fallback if not enough samples meet criteria
    if len(inputs_idx) < n_input_t:
        inputs_idx = [pdx for pdx in range(len(coverage)) if pdx != cloudless_idx][:n_input_t]
        coverage_match = False
    else:
        coverage_match = True

    # Verify target sample meets clearness criteria
    if coverage[cloudless_idx] > clear_tresh:
        coverage_match = False

    return inputs_idx, cloudless_idx, coverage_match


def fixedsubset_sampler(n_input_t: int, min_cov: float, max_cov: float, coverage: CoverageArray, clear_tresh: float = 1e-3, earliest_idx: int = 0, latext_idx: int = 30) -> Tuple[IndexList, int, bool]:
    """
    Sample time points from a subsequence with fixed coverage criteria.

    Applies fixed_sampler to a subsequence, falling back to full sequence if criteria
    aren't met in the subsequence.

    Args:
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold
        max_cov: Maximum cloud coverage threshold
        coverage: Array of cloud coverage values
        clear_tresh: Maximum coverage threshold for target sample
        earliest_idx: Start index of subsequence
        latext_idx: End index of subsequence

    Returns:
        Tuple containing input indices, target index, and coverage match flag
    """
    # Apply fixed sampler to subsequence
    inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage[earliest_idx:latext_idx], clear_tresh)

    # Adjust indices for subsequence offset
    inputs_idx = [idx + earliest_idx for idx in inputs_idx]
    cloudless_idx += earliest_idx

    # Fallback to full sequence if criteria not met
    if not coverage_match:
        inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)

    return inputs_idx, cloudless_idx, coverage_match


def sampler(
    sampling: SamplingMethod,
    n_input_t: int,
    min_cov: float,
    max_cov: float,
    coverage: CoverageArray,
    clear_tresh: float = 1e-3,
    earliest_idx: int = 0,
    latext_idx: int = 30,
) -> Tuple[IndexList, int, bool]:
    """
    Main sampling function that routes to specific sampling strategies.

    Args:
        sampling: Sampling method ('random', 'fixedsubset', or 'fixed')
        t_windows: List of temporal windows (for random sampling)
        n_input_t: Number of input samples to select
        min_cov: Minimum cloud coverage threshold
        max_cov: Maximum cloud coverage threshold
        coverage: Array of cloud coverage values
        clear_tresh: Maximum coverage threshold for target sample
        earliest_idx: Start index for subsequence sampling
        latext_idx: End index for subsequence sampling

    Returns:
        Tuple containing input indices, target index, and coverage match flag
    """
    if sampling == "fixedsubset":
        return fixedsubset_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh, earliest_idx, latext_idx)
    else:  # Default to fixed sampling
        return fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)
