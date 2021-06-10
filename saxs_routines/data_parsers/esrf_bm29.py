"""
Data parsers for SAXS performed on the BM29 beamline at the ESRF with
a Pilatus detector.

"""

import numpy as np
import h5py

from saxs_routines.sample import Sample


def readHPLC(filepath, **sample_kwargs):
    """Reader for an experiment with HPLC in HDF format.

    Parameters
    ----------
    filepath : str
        The path of the file to be read
    sample_kwargs : dict, optional
        Additional keywords to be passed to the
        :py:class:`saxs_routines.sample.Sample` class after file reading.

    """
