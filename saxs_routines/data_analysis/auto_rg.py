"""This module provides a routine to compute the radius of gyration.

"""


import numpy as np

import matplotlib.pyplot as plt

from saxs_routines.models.builtins import model_linear_rg


class AutoRg:
    """Computes a radius of gyration for the provided data.

    The methods consists in fitting a radius of gyration (Rg) and an intensity
    at the limit of :math:`\\rm q = 0~ \\mathring{A}^{-1}` (I0). This is done
    by considering several block sizes that will slide along the q range.
    For each block position, a fit is performed.
    Subsequently, the distribution of Rg and I0 are computed and the mean
    values for these two parameters is obtained from the distribution.

    The class can treat both 1D and 2D datasets.

    .. note::
        By default, the class uses a linear model to fit the Rg, such
        that the user should provide the logarithm of the data if no
        user-defined model is provided.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        An instance of :py:class:`sample.Sample` containing the data to fit.
        No transformation is performed on the data except slicing. Hence, if
        the model requires to use the log of the data, it is up to the user
        to apply the operator before passing the data the the class.
    qmin : float, optional
        Minimum value for the range of q values to be used.
        (default 0.005)
    qmax : float, optional
        Maximum value for the range of q values to be used.
        (default 0.2)
    min_n : int, optional
        Minimum number of points to use for the fit.
        (default 10)
    qRg_limits : 2-tuple of floats, optional
        Minimum and maximum limit of q*Rg value to consider the fit valid.
        (default (0.0005, 1.3))
    model : :py:class:`model.Model`
        An instance of Model class to be used to fit the data.
        If none, a built-in model from SAXS_routines will be used.
    fit_kws : dict, optional
        Additional keywords to pass the fit function
        (default None)

    Attributes
    ----------
    fitResult : list of :py:class:`model.Model`
        A list of fitted models, each being an instance of the
        :py:class:`model.Model` class.
    prob : list of np.array
        A list of arrays giving the posterior probability that the
        Rg and I0 values are correct given the data as a function of
        Rg and I0. There is one array per 1D scattering curve in the
        dataset.
    rg : list of float
        The values of Rg determined by the routine for each 1D scattering
        curve in the dataset.
    rg_std : list of float
        The errors for Rg determined by the routine for each 1D scattering
        curve in the dataset.
    I0 : list of float
        The values of I0 determined by the routine for each 1D scattering
        curve in the dataset.
    I0_std : list of float
        The errors for I0 determined by the routine for each 1D scattering
        curve in the dataset.

    Examples
    --------
    The class can be initialized from a 2D dataset using:

    >>> import numpy as np
    >>> from saxs_routines.data_parsers.esrf_bm29 import read_HPLC
    >>> from saxs_routines.data_analysis.auto_rg import AutoRg
    >>>
    >>> data = read_HPLC('myFile', name='myProtein')
    >>> # buffer subtraction
    >>> sub = data[480:520] - data[:50].mean(0)
    >>>
    >>> # AutoRg part
    >>> autorg = AutoRg(np.log(sub))
    >>> autorg.run()
    >>> autorg.rg
    [6.625, 6.232, 6.457, ..., 6.890, 7.011]

    The result can be plotted as well as other attributes:

    >>> autorg.plot()

    """

    def __init__(
        self,
        data,
        qmin=0.005,
        qmax=0.25,
        min_n=4,
        qRg_limits=(0.1, 1.3),
        model=None,
        fit_kws=None,
    ):
        self.data = self._clean_data(data)
        self.qmin = qmin
        self.qmax = qmax
        self.min_n = min_n
        self.qRg_limits = qRg_limits

        self.model = model
        if self.model is None:
            self.model = model_linear_rg()

        self.fit_kws = {}
        if fit_kws is not None:
            self.fit_kws.update(fit_kws)

        self.fitResult = []
        self.prob = []
        self.rg = []
        self.rg_std = []
        self.I0 = []
        self.I0_std = []

    def run(self):
        """Run the algorithm to extract the radius of gyration.

        The result is stored in the *fitResult*, *prob*,
        *rg*, *I0*, *rg_std* and *I0_std* attributes.

        """
        self.fitResult = []
        self.prob = []
        self.rg = []
        self.rg_std = []
        self.I0 = []
        self.I0_std = []

        self._process_dimensions()

    def _process_dimensions(self):
        data = self.data
        if data.ndim == 1:
            self._fit_rg(data)

        elif data.ndim == 2:
            if data.shape[0] == data.q.size:
                data = data.T
            for val in data:
                self._fit_rg(val)

        else:
            print("Too many dimensions, the method can treat data up to 2D.")
            return

    def _fit_rg(self, data):
        reduced = data.get_q_range(self.qmin, self.qmax)
        res = []
        prob = []
        p_norm = reduced.max() * (
            np.min(self.qRg_limits[1] / reduced.q)
            - np.max(self.qRg_limits[0] / reduced.q)
        ).view(np.ndarray)
        for block_size in np.arange(self.min_n, reduced.size, 2):
            for start in np.arange(
                0, reduced.size - block_size, int(block_size / 2)
            ):
                q_start = reduced.q[start]
                q_end = reduced.q[start + block_size]
                sel = reduced.get_q_range(q_start, q_end)
                try:
                    self.model.fit(
                        sel.q, sel, weights=sel.errors, **self.fit_kws
                    )
                    if self._validate_rg(
                        sel.q, self.model.optParams["rg"].value
                    ):
                        res.append(self.model.copy())
                except RuntimeError:
                    pass

                # compute the bayesian likelihood
                chi = (sel - self.model.eval(sel.q, self.model.optParams)) ** 2
                chi /= sel.errors**2
                chi = np.sum(chi[np.isfinite(chi)]).view(np.ndarray)
                denominator = np.sqrt(np.linalg.det(self.model.fitResult[1]))
                prob.append(
                    4 * np.pi * np.exp(-chi / 2) / (denominator * p_norm)
                )

        rg = np.histogram([val.optParams["rg"].value for val in res], 50)
        rg_std = np.histogram([val.optParams["rg"].error for val in res], 50)
        I0 = np.histogram([val.optParams["I0"].value for val in res], 50)
        I0_std = np.histogram([val.optParams["I0"].error for val in res], 50)

        self.rg.append(np.sum(rg[0] * rg[1][:-1]) / np.sum(rg[0]))
        self.I0.append(np.sum(I0[0] * I0[1][:-1]) / np.sum(I0[0]))
        self.rg_std.append(
            np.sum(rg_std[0] * rg_std[1][:-1]) / np.sum(rg_std[0])
        )
        self.I0_std.append(
            np.sum(I0_std[0] * I0_std[1][:-1]) / np.sum(I0_std[0])
        )
        self.fitResult.append(res)
        self.prob.append(prob)

    def _validate_rg(self, q, rg):
        return ~np.any(q * rg < self.qRg_limits[0]) & ~np.any(
            q * rg > self.qRg_limits[1]
        )

    def _clean_data(self, data):
        np.place(data.errors, ~np.isfinite(data), np.inf)
        np.place(data, ~np.isfinite(data), 0)

        return data

    def plot(self, plot_errors=True, new_fig=False):
        """Plot the fitted parameters, Rg and I0.

        Initially intended to be used with 2D dataset with time
        as first axis.

        Parameters
        ----------
        plot_errors : bool, optional
            Whether to plot the errors on the parameters or not.
            (default False)
        new_fig : bool, optional
            If True, a new figure will be created. Otherwise, use
            the existing one if any.
            (default False)

        """
        if new_fig or len(plt.get_fignums()) == 0:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.subplots(2, 1)
        else:
            fig = plt.gcf()
            ax = fig.axes

        time = self.data.time
        err_rg = np.array(self.rg_std).copy()
        np.place(err_rg, ~np.isfinite(err_rg), 0)
        err_I0 = np.array(self.I0_std).copy()
        np.place(err_I0, ~np.isfinite(err_I0), 0)
        if plot_errors is False:
            err_rg *= 0.0
            err_I0 *= 0.0

        ax[0].plot(time, self.rg, label=self.data.name)
        ax[0].fill_between(time, self.rg - err_rg, self.rg + err_rg, alpha=0.2)
        ax[0].set_ylabel("Rg [nm]")
        ax[1].plot(time, self.I0, label=self.data.name)
        ax[1].fill_between(time, self.I0 - err_I0, self.I0 + err_I0, alpha=0.2)
        ax[1].set_ylabel("I0 [A.U.]")
        ax[1].set_xlabel("Time [min]")
        ax[0].legend()

    def plot_fit(self, idx=0, best=False, new_fig=True):
        """Plot the fitted model against experimental data.

        Parameters
        ----------
        idx : int
            Index of the data to be plotted for 2D dataset of
            time series.
        best : bool, optional
            If True, plot only the best estimate of the Rg and I0.
            If False, plot all fitted models.
            (default, False)

        """
        data = self.data.get_q_range(self.qmin, self.qmax)
        if data.shape[0] == data.q.size:
            data = data.T

        if data.ndim == 2:
            data = data[idx]

        fig, ax = data.plot("q2", new_fig=new_fig)
        if best:
            for idx, val in enumerate(self.rg):
                popt = np.mean(
                    [
                        val.optParams.paramList[0]
                        for val in self.fitResult[idx]
                    ],
                    0,
                )
                plt.plot(
                    data.q**2,
                    self.model.eval(data.q, popt),
                    color="tab:orange",
                    ls=":",
                )
        else:
            for idx, val in enumerate(self.fitResult[idx]):
                plt.plot(
                    data.q**2,
                    self.model.eval(data.q, val.optParams),
                    color="tab:orange",
                    ls=":",
                )

        return fig, ax

    def plot_rg_posterior_prob(self, idx=0):
        """Plot the computed posterior probability for the Rg value.

        The posterior probability is computed from the Bayes rule for
        each fit at a given index.
        The plot gives the probability as a function of fitted radius of
        gyration.

        Parameters
        ----------
        idx : int
            Index of the data to be plotted for 2D dataset of
            time series.

        """
        fig, ax = plt.subplots()
        res = self.fitResult[idx]
        vals = np.array([val.optParams["rg"].value for val in res])
        prob = np.array(self.prob[idx])
        sort_ids = np.argsort(vals)
        ax.plot(vals[sort_ids], prob[sort_ids])
        ax.set_xlabel("Rg [nm]")
        ax.set_ylabel("P(Rg, I0 | data)")

    def plot_I0_posterior_prob(self, idx=0):
        """Plot the computed posterior probability for the I0 value.

        The posterior probability is computed from the Bayes rule for
        each fit at a given index.
        The plot gives the probability as a function of fitted I0.

        Parameters
        ----------
        idx : int
            Index of the data to be plotted for 2D dataset of
            time series.

        """
        fig, ax = plt.subplots()
        res = self.fitResult[idx]
        vals = np.array([val.optParams["I0"].value for val in res])
        prob = np.array(self.prob[idx])
        sort_ids = np.argsort(vals)
        ax.plot(vals[sort_ids], prob[sort_ids])
        ax.set_xlabel("I0 [A.U.]")
        ax.set_ylabel("P(Rg, I0 | data)")
