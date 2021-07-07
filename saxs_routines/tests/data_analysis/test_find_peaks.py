from saxs_routines.data_analysis import FindPeaks


def test_find_peaks(bm29_HPLC_hdf):
    peaks = FindPeaks(bm29_HPLC_hdf)
    peaks.run()

    assert peaks.peaks[0] == 990
    assert peaks.get_sub_arrays()[0].shape == (168, 1043)
