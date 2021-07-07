"""Automatic peak finding for 2D SEC-SAS data.

"""

from scipy.signal import find_peaks, peak_widths


class FindPeaks:
    """Finds the peaks in a time series from a 2D SEC-SAS dataset.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        An instance of :py:class:`sample.Sample` containing
        a 2D dataset where the first axis is assumed to be the elution time
        and the second axis is assumed to be the momentum transfer q values.
    height_factor : float
        A float between 0 and 1. This will be used to define the minimum height
        of the peak by taking *height = height_factor * data.max()*.

    """

    def __init__(self, data, height_factor=0.5):
        self.data = data
        self.height_factor = height_factor

        self.peaks = None
        self.widths = None
        self.heights = None
        self.l_borders = None
        self.r_borders = None

    def run(self, find_peaks_kws=None, peak_widths_kws=None):
        """Run the algorithm to find the peaks and associated widths.

        Parameters
        ----------
        find_peaks_kws : dict
            Additional keywords to be passed to scipy *find_peaks* keywords.
        peak_widths_kws : dict
            Additional keywords to be passed to scipy *peak_widths* keywords.

        """
        if find_peaks_kws is None:
            find_peaks_kws = {}

        if peak_widths_kws is None:
            peak_widths_kws = {}

        fp_kws = {"height": float(self.data.sum(1).max())}
        fp_kws.update(find_peaks_kws)
        self._get_peaks(**fp_kws)
        self._get_widths(**peak_widths_kws)

    def get_sub_arrays(self):
        """Return the array corresponding to the region within peak borders."""
        out = []
        x = self.data.time
        for idx, peak in enumerate(self.peaks):
            left = x[int(self.l_borders[idx])]
            right = x[int(self.r_borders[idx])]
            out.append(self.data.get_time_range(left, right))

        return out

    def _get_peaks(self, **kwargs):
        """Find the peaks in the time series."""
        self.peaks = find_peaks(
            self.data.sum(1), height=float(self.data.sum(1).max())
        )[0]

    def _get_widths(self, **kwargs):
        """Find the widths and associated limits for each peak."""
        res = peak_widths(self.data.sum(1), self.peaks)
        self.widths, self.heights, self.l_borders, self.r_borders = res
