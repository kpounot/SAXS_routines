"""
Data parsers for SAXS performed on the BM29 beamline at the ESRF with
a Pilatus detector.

"""

import numpy as np
import h5py

from saxs_routines.sample import Sample


def read_HPLC(filepath, **sample_kwargs):
    """Reader for an experiment with HPLC in HDF format.

    Parameters
    ----------
    filepath : str
        The path of the file to be read
    sample_kwargs : dict, optional
        Additional keywords to be passed to the
        :py:class:`saxs_routines.sample.Sample` class after file reading.

    """
    data = h5py.File(filepath, "r")

    intensities = data["scattering_I"][()]
    errors = data["scattering_Stdev"][()]
    I0 = data["I0"][()]
    rg = data["Rg"][()]
    q = data["q"][()]
    time = data["time"][()]
    elution_volume = data["volume"][()]

    out = Sample(
        intensities,
        errors=errors,
        I0=I0,
        rg=rg,
        q=q,
        time=time,
        elution_volume=elution_volume,
        beamline="ESRF - BM29",
    )

    out.__dict__.update(sample_kwargs)

    return out
