import pytest

import numpy as np

from saxs_routines.models.builtins import model_linear_rg


def test_fit_rg_linear(bm29_HPLC_hdf):
    data = bm29_HPLC_hdf - bm29_HPLC_hdf[:20].mean(0)
    data = data.get_q_range(0.04, 0.1)

    m = model_linear_rg()

    m.fit(data.q, np.log(data)[1000], weights=np.log(data).errors[1000])

    assert (
        np.sum((np.log(data)[1000] - m.eval(data.q, **m.optParams)) ** 2) < 0.4
    )
