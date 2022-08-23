"""Data processing functions to be used prior to analysis."""


import numpy as np

from saxs_routines import Sample


def average_frames(frames, max_chi_square=None, verbose=True):
    """Average multiple frames.

    Parameters
    ----------
    frames : list of :py:class:`Sample` instances
        The frames to be averaged.
    max_chi_square : float, optional
        Maximum chi-square between individual frames and the average.
        Frames with higher chi-square will be discarded.
        (default, None, will be inferred using the chi-square distribution
        variance : 2 * number of data points)
    verbose : bool, optional
        If True, prints information about the chi-square filtering.
        (default, True)

    """
    avg = np.mean(frames, 0)

    if max_chi_square is None:
        max_chi_square = 2 * frames[0].q.size

    chi = []
    filt_frames = []
    for frame in frames:
        chi.append(np.sum((frame - avg) ** 2 / frame.errors**2))
        if chi[-1] <= max_chi_square:
            filt_frames.append(frame)

    avg = np.mean(filt_frames, 0)
    err = np.sqrt(
        np.sum([frame.errors**2 for frame in filt_frames], 0)
    ) / len(filt_frames)

    if verbose:
        print(
            f"Averaging using max_chi_square={max_chi_square}; "
            + f"keeping {len(filt_frames)} frames out of {len(frames)}."
        )

    sample = Sample(avg)
    sample.__dict__.update(frames[0].__dict__)
    sample.errors = err

    return sample
