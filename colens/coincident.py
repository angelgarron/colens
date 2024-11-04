"Functions related to the coincident analysis."
import numpy as np


def get_coinc_indexes(idx_dict, time_delay_idx, instruments):
    """Return the indexes corresponding to coincident triggers, requiring
    they are seen in at least two detectors in the network

    Parameters
    ----------
    idx_dict: dict
        Dictionary of indexes of triggers above threshold in each
        detector
    time_delay_idx: dict
        Dictionary giving time delay index (time_delay*sample_rate) for
        each ifo

    Returns
    -------
    coinc_idx: list
        List of indexes for triggers in geocent time that appear in
        multiple detectors
    """
    coinc_list = np.array([], dtype=int)
    for ifo in instruments:
        # Create list of indexes above threshold in single detector in geocent
        # time. Can then search for triggers that appear in multiple detectors
        # later.
        if len(idx_dict[ifo]) != 0:
            coinc_list = np.hstack([coinc_list, idx_dict[ifo] - time_delay_idx[ifo]])
    # Search through coinc_idx for repeated indexes. These must have been loud
    # in at least 2 detectors.
    counts = np.unique(coinc_list, return_counts=True)
    coinc_idx = counts[0][counts[1] > 1]
    return coinc_idx
