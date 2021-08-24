import os

import pytest

from saxs_routines.data_parsers import esrf_bm29 as bm29


path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def bm29_HPLC_hdf():
    return bm29.read_HPLC(path + "/sample_data/WT_001.h5", name="wt")


@pytest.fixture
def bm29_1d():
    return bm29.read_processed_1d(path + "/sample_data/WT_001_00001.dat")


autorg_plot_files = [
    path + "/data_analysis/autorg_plot.csv",
    path + "/data_analysis/autorg_plot_fit.csv",
    path + "/data_analysis/autorg_plot_I0_posterior.csv",
    path + "/data_analysis/autorg_plot_rg_posterior.csv",
]
