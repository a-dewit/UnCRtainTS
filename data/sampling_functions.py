import numpy as np


def fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh=1e-3):
    # sample custom time points from the current patch space in the current split
    # sort observation indices according to cloud coverage, ascendingly
    coverage_idx = np.argsort(coverage)
    cloudless_idx = coverage_idx[0]  # take the (earliest) least cloudy sample
    # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
    inputs_idx = [pdx for pdx, perc in enumerate(coverage) if perc >= min_cov and perc <= max_cov][:n_input_t]
    if len(inputs_idx) < n_input_t:
        # ... if not exists then take the first n_input_t samples (except target patch)
        inputs_idx = [pdx for pdx in range(len(coverage)) if pdx != cloudless_idx][:n_input_t]
        coverage_match = False  # flag input samples that didn't meet the required cloud coverage
    else:
        coverage_match = True  # assume the requested amount of cloud coverage is met
    # check whether the target meets the requested amount of clearness
    if coverage[cloudless_idx] > clear_tresh:
        coverage_match = False
    return inputs_idx, cloudless_idx, coverage_match


def fixedsubset_sampler(n_input_t, min_cov, max_cov, coverage, earliest_idx=0, latext_idx=30, clear_tresh=1e-3):
    # apply the fixed sampler on only a subsequence of the input sequence
    inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage[earliest_idx:latext_idx], clear_tresh)
    # shift sampled indices by the offset of the subsequence
    inputs_idx, cloudless_idx = (
        [idx + earliest_idx for idx in inputs_idx],
        cloudless_idx + earliest_idx,
    )
    # if the sampled indices do not meet the criteria, then default to sampling over the full time series
    if not coverage_match:
        inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)
    return inputs_idx, cloudless_idx, coverage_match


def random_sampler(t_windows, coverage, clear_tresh=1e-3):
    # sample a random target time point below 0.1% coverage (i.e. coverage<1e-3), or at min coverage
    is_clear = np.argwhere(np.array(coverage) < clear_tresh).flatten()
    try:
        cloudless_idx = is_clear[np.random.randint(0, len(is_clear))]
    except Exception:
        cloudless_idx = np.array(coverage).argmin()
    # around this target time point, pick self.n_input_t input time points
    windows = [window for window in t_windows if cloudless_idx in window]
    # we pick the window with cloudless_idx centered such that input samples are temporally adjacent,
    # alternatively: pick a causal window (with cloudless_idx at the end) or randomly sample input dates
    inputs_idx = [input_t for input_t in windows[len(windows) // 2] if input_t != cloudless_idx]
    coverage_match = True  # note: not checking whether any requested cloud coverage is met in this mode
    return inputs_idx, cloudless_idx, coverage_match


def sampler(sampling, t_windows, n_input_t, min_cov, max_cov, coverage, clear_tresh=1e-3, earliest_idx=0, latext_idx=30):
    if sampling == "random":
        inputs_idx, cloudless_idx, coverage_match = random_sampler(t_windows, coverage, clear_tresh)
    elif sampling == "fixedsubset":
        inputs_idx, cloudless_idx, coverage_match = fixedsubset_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh, earliest_idx=earliest_idx, latext_idx=latext_idx)
    else:  # default to fixed sampler
        inputs_idx, cloudless_idx, coverage_match = fixed_sampler(n_input_t, min_cov, max_cov, coverage, clear_tresh)

    return inputs_idx, cloudless_idx, coverage_match
