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
        :py:class:`sample.Sample` class after file reading.

    """
    data = h5py.File(filepath, "r")

    intensities = data["scattering_I"][()]
    errors = data["scattering_Stdev"][()]
    I0 = data["I0"][()]
    I0_std = data["I0_Stdev"][()]
    rg = data["Rg"][()]
    rg_std = data["Rg_Stdev"][()]
    q = data["q"][()]
    time = data["time"][()]
    elution_volume = data["volume"][()]

    out = Sample(
        intensities,
        filename=filepath,
        errors=errors,
        I0=I0,
        I0_std=I0_std,
        rg=rg,
        rg_std=rg_std,
        q=q,
        time=time,
        elution_volume=elution_volume,
        beamline="ESRF - BM29",
    )

    out.__dict__.update(sample_kwargs)

    return out


def read_processed_1d(filepath, **sample_kwargs):
    """Reader for an experiment with HPLC in HDF format.

    Parameters
    ----------
    filepath : str
        The path of the file to be read
    sample_kwargs : dict, optional
        Additional keywords to be passed to the
        :py:class:`sample.Sample` class after file reading.

    """
    with open(filepath) as data_file:
        data = data_file.readlines()

    info = {}
    for idx, line in enumerate(data):
        if "Detector" in line:
            info["detector"] = line.split(" = ")[1]
        if "Wavelength" in line:
            info["wavelength"] = line.split(" = ")[1]
        if "Measurement Temperature" in line:
            info["temperature"] = line.split(":")[1]
        if "Concentration" in line:
            info["concentration"] = line.split(":")[1]
        if "#" not in line:
            break

    data = np.loadtxt(filepath)

    out = Sample(
        data[:, 1],
        q=data[:, 0],
        errors=data[:, 2],
        beamline="ESRF - BM29",
        filename=filepath,
        **info
    )

    out.__dict__.update(sample_kwargs)

    return out


def read_abs_data(filepath):
    """Reader for text file of absorption data from Shimadzu HPLC.

    Parameters
    ----------
    filepath : str
        The path of the file to be read.

    Returns
    -------
    times : np.array
        The elution time in minutes.
    wavelengths : np.array
        The wavelengths at which absorption is measured in nm.
    dataset : np.ndarray
        The 2D dataset of the absorption data with time along the
        first axis and wavelength along the second.

    """
    with open(filepath) as f:
        data = f.read().splitlines()

    data_idx = 0
    for idx, l in enumerate(data):
        if "[PDA 3D]" in l:
            data_idx = idx
            break

    tint = float(data[data_idx + 1].split("\t")[1])
    tinit = float(data[data_idx + 2].split("\t")[1])
    tend = float(data[data_idx + 3].split("\t")[1])
    times = np.arange(tinit, tend, tint / (60 * 1000))

    winit = float(data[data_idx + 4].split("\t")[1])
    wend = float(data[data_idx + 5].split("\t")[1])
    wnum = float(data[data_idx + 6].split("\t")[1])
    wavelengths = np.arange(winit, wend, (wend - winit) / wnum)

    dataset = []
    for idx in range(times.size):
        line = data[data_idx + 10 + idx].split("\t")
        dataset.append(np.array(line[1:]).astype(float))

    return times, wavelengths, np.array(dataset)
