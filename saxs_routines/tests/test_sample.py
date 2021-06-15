import numpy as np

import pytest


ufunc_list = (
    np.add,
    np.multiply,
    np.subtract,
    np.divide,
    np.power,
)

expect_list = (
    855.8046,
    183100.39,
    0.0,
    1.0,
    np.inf,
)


@pytest.mark.parametrize("ufunc,expect", zip(ufunc_list, expect_list))
def test_ufuncsi_double_op(bm29_HPLC_hdf, ufunc, expect):
    res = ufunc(bm29_HPLC_hdf, bm29_HPLC_hdf)
    np.testing.assert_allclose(res[5, 5], expect)
    assert np.any(res.errors != bm29_HPLC_hdf.errors)


ufunc_list = (
    np.sqrt,
    np.cbrt,
    np.square,
    np.log,
    np.exp,
)

expect_list = (
    20.6858,
    7.5355487,
    183100.39,
    6.058895,
    np.inf,
)


@pytest.mark.parametrize("ufunc,expect", zip(ufunc_list, expect_list))
def test_ufuncs_single_op(bm29_HPLC_hdf, ufunc, expect):
    res = ufunc(bm29_HPLC_hdf)
    np.testing.assert_allclose(res[5, 5], expect)
    assert np.any(res.errors != bm29_HPLC_hdf.errors)


def test_sample_bin(bm29_HPLC_hdf):
    assert bm29_HPLC_hdf[:20].bin(2).shape == (10, 1043)


def test_sample_sliding_average(bm29_HPLC_hdf):
    assert bm29_HPLC_hdf[:20].sliding_average(10).shape == (10, 1043)


def test_get_q_range(bm29_1d):
    assert bm29_1d.get_q_range(0.03, 0.1).q.size == 14
