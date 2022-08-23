"""Common functions for data parsing."""


import numpy as np

from saxs_routines import Sample


def read_text(filepath, sample_kwargs=None, loadtxt_kwargs=None):
    """Read data from a text file.

    It is assumed that data are given as three columns corresponding to
    momentum transfer, intensity and errors.

    Parameters
    ----------
    filepath : str
        The path of the file to be opened.
    sample_kwargs : dict, optional
        Additional arguments to pass to :py:class:`Sample` class constructor.
        (default, None)
    loadtxt_kwargs : dict, optional
        Additional arguments to pass to the `numpy.loadtxt` routine.

    Returns
    -------
    sample : :py:class:`Sample`
        An instance of the :py:class:`Sample` class containing the loaded data.

    """
    loadtxt_kwargs = {} if loadtxt_kwargs is None else loadtxt_kwargs
    sample_kwargs = {} if sample_kwargs is None else sample_kwargs

    data = np.loadtxt(filepath, **loadtxt_kwargs)

    intensities = data[:, 1]
    errors = data[:, 2]
    q = data[:, 0]

    out = Sample(
        intensities,
        filename=filepath,
        errors=errors,
        q=q,
    )

    out.__dict__.update(sample_kwargs)

    return out
