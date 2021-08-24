import os

import numpy as np

import pytest

import matplotlib.pyplot as plt

from saxs_routines.data_analysis import AutoRg


path = os.path.dirname(os.path.abspath(__file__))


def test_autorg(bm29_HPLC_hdf):
    sample = bm29_HPLC_hdf.get_time_range(480, 520).bin(8, metadata=["time"])
    sample -= bm29_HPLC_hdf.get_time_range(200, 300).mean(0)
    rg = AutoRg(np.log(sample), min_n=7)
    rg.run()

    rg.plot()
    ref = np.loadtxt(path + "/autorg_plot.csv")
    dat = plt.gca().lines[-1].get_xydata()
    np.testing.assert_array_almost_equal(ref, dat)

    rg.plot_fit()
    ref = np.loadtxt(path + "/autorg_plot_fit.csv")
    dat = plt.gca().lines[-1].get_xydata()
    np.testing.assert_array_almost_equal(ref, dat)

    rg.plot_I0_posterior_prob()
    ref = np.loadtxt(path + "/autorg_plot_I0_posterior.csv")
    dat = plt.gca().lines[-1].get_xydata()
    np.testing.assert_array_almost_equal(ref, dat)

    rg.plot_rg_posterior_prob()
    ref = np.loadtxt(path + "/autorg_plot_rg_posterior.csv")
    dat = plt.gca().lines[-1].get_xydata()
    np.testing.assert_array_almost_equal(ref, dat)
