import os

import pytest

from saxs_routines.data_parsers import esrf_bm29 as bm29


path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def bm29_HPLC_hdf():
    return bm29.read_HPLC(path + "/sample_data/WT_001.h5", name="wt")
