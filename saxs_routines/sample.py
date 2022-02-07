"""
Handle data associated with a sample.

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

from saxs_routines.range_selector import RangeSelector


class Sample(np.ndarray):
    """Handle the measured data along with metadata.

    This class is a subclass of the numpy.ndarray class with additional
    methods and attributes that are specific to small-angle
    X-ray scattering experiments on biological samples.

    It can handle various operations such as addition and subtraction
    of sample data or numpy array, scaling by a scalar or an array,
    indexing, broadcasting, reshaping, binning, sliding average or
    data cleaning.

    Parameters
    ----------
    input_arr : np.ndarray, list, tuple or scalar
        Input array corresponding to sample scattering data.
    kwargs : dict (optional)
        Additional keyword arguments either for :py:meth:`np.asarray`
        or for sample metadata. The metadata are:
            - **filename**, the name of the file used to extract the data.
            - **errors**, the errors associated with scattering data.
            - **time**, the experimental time.
            - **elution_volume**, if available, the elution volume of the HPLC.
            - **I0**, the fitted intensity at q = 0.
            - **I0_std**, the uncertainty on I0 value(s).
            - **rg**, the gyration radius.
            - **rg_std**, the uncertainty on Rg value(s).
            - **wavelength**, the wavelength of the X-ray beam.
            - **name**, the name for the sample.
            - **temperature**, the temperature(s) used experimentally.
            - **concentration**, the concentration of the sample.
            - **pressure**, the pressure used experimentally.
            - **buffer**, a description of the buffer used experimentally.
            - **q**, the values for the momentum transfer q.
            - **detector**, the detector used.
            - **beamline**, the name of the beamline used.
            - **flow_rate**, the flow rate inside the capillary.
            - **observable**, for 2D dataset the name of the attribute
                corresponding to the first axis.

    Note
    ----
    The **errors** metadata is special as it is updated for various operations
    that are performed on the data array such as indexing or for the use
    of universal functions.
    For instance, indexing of the data will be performed on **errors** as
    well if its shape is the same as for the data. Also, addition,
    subtraction and other universal functions will lead to automatic error
    propagation.
    Some other metadata might change as well, like **q**, but only for
    the use of methods specific of the :py:class:`Sample` class and
    not for methods inherited from numpy.

    Examples
    --------
    A sample can be created using the following:

    >>> s1 = Sample(
    ...     np.arange(5),
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.2, 0.12, 0.14, 0.15])
    ... )

    >>> buffer = Sample(
    ...     [0., 0.2, 0.4, 0.3, 0.1],
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.2, 0.05, 0.1, 0.2])
    ... )

    where *my_data*, *my_errors* and *q_values* are numpy arrays.
    A buffer subtraction can be performed using:

    >>> s1 = s1 - buffer
    Sample([0. , 0.80000001, 1.60000002, 2.70000005, 3.9000001], dtype=float32)

    where *buffer1* is another instance of :py:class:`Sample`. The error
    propagation is automatically performed and the other attributes are taken
    from the first operand (here s1).
    Other operations such as scaling can be performed using:

    >>> s1 = 0.8 * s1
    Sample([0. , 0.80000001, 1.60000002, 2.4000001, 3.20000005], dtype=float32)

    You can transform another :py:class:`Sample` instance into a column
    vector and look how broadcasting and error propagation work:

    >>> s2 = Sample(
    ...     np.arange(5, 10),
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.3, 0.05, 0.1, 0.2])
    ... )
    >>> s2 = s2[:, np.newaxis]
    >>> res = s1 * s2
    >>> res.errors
    array([[0.5       , 1.00498756, 0.63245553, 0.76157731, 0.85      ],
           [0.6       , 1.23693169, 0.93722996, 1.23109707, 1.5       ],
           [0.7       , 1.40089257, 0.84593144, 0.99141313, 1.06887792],
           [0.8       , 1.60312195, 0.98061205, 1.15948264, 1.26491106],
           [0.9       , 1.81107703, 1.1516944 , 1.3955644 , 1.56923548]])

    """

    def __new__(cls, input_arr, **kwargs):
        if not isinstance(input_arr, Sample):
            obj = np.asarray(
                input_arr,
                **{
                    key: val
                    for key, val in kwargs.items()
                    if key in ("dtype", "order")
                }
            ).view(cls)
        else:
            obj = input_arr

        for key, val in kwargs.items():
            if key not in ("dtype", "order"):
                setattr(obj, key, val)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.errors = getattr(obj, "errors", 0)
        self.time = getattr(obj, "time", 0)
        self.elution_volume = getattr(obj, "elution_volume", 0)
        self.absorbance_time = getattr(obj, "absorbance_time", 0)
        self.absorbance = getattr(obj, "absorbance", 0)
        self.absorbance_wavelength = getattr(obj, "absorbance_wavelength", 0)
        self.I0 = getattr(obj, "I0", 0)
        self.I0_std = getattr(obj, "I0_std", 0)
        self.rg = getattr(obj, "rg", 0)
        self.rg_std = getattr(obj, "rg_std", 0)
        self.wavelength = getattr(obj, "wavelength", 0)
        self.filename = getattr(obj, "filename", 0)
        self.name = getattr(obj, "name", self.filename)
        self.temperature = getattr(obj, "temperature", 0)
        self.concentration = getattr(obj, "concentration", 0)
        self.pressure = getattr(obj, "pressure", 0)
        self.buffer = getattr(obj, "buffer", 0)
        self.q = getattr(obj, "q", 0)
        self.detectors = getattr(obj, "detectors", 0)
        self.beamline = getattr(obj, "beamline", 0)
        self.flow_rate = getattr(obj, "flow_rate", 0)
        self.observable = getattr(obj, "observable", "time")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inp_cast = []
        out_cast = []
        for inp in inputs:
            if isinstance(inp, Sample):
                inp_cast.append(inp.view(np.ndarray))
            else:
                inp_cast.append(inp)

        if "out" in kwargs.keys():
            for out in kwargs["out"]:
                if isinstance(out, Sample):
                    out_cast.append(out.view(np.ndarray))
                else:
                    out_cast.append(out)
            kwargs["out"] = tuple(out_cast)

        obj = super().__array_ufunc__(ufunc, method, *inp_cast, **kwargs)
        if obj is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            obj = [
                obj,
            ]

        obj[0] = self._process_attributes(
            obj[0], ufunc, method, *inputs, **kwargs
        )

        return obj[0] if ufunc.nout == 1 else tuple(obj)

    def _process_attributes(self, obj, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs.keys():
            kwargs["out"] = (None,) * ufunc.nout

        errors = [Sample(inp).errors for inp in inputs]
        inp_cast = []
        inp_dict = []
        for idx, inp in enumerate(inputs):
            if isinstance(inp, Sample):
                inp_dict.append(inp.__dict__)
            inp_cast.append(np.asarray(inp))

        obj = np.asarray(obj).view(Sample)
        obj.__dict__.update(inp_dict[0])

        if ufunc == np.add or ufunc == np.subtract:
            if method == "__call__":
                obj.errors = np.sqrt(
                    np.add(
                        np.power(errors[0], 2),
                        np.power(errors[1], 2),
                        **kwargs
                    ),
                )
            elif method == "reduce":
                obj.errors = np.sqrt(
                    np.add.reduce(np.power(errors[0], 2), **kwargs),
                )

        elif ufunc == np.multiply:
            if method == "__call__":
                obj.errors = np.sqrt(
                    np.add(
                        np.power(inp_cast[1] * errors[0], 2),
                        np.power(inp_cast[0] * errors[1], 2),
                        **kwargs
                    ),
                )
            elif method == "reduce":
                obj.errors = np.sqrt(
                    np.multiply.reduce(
                        np.power(inp_cast[0] * errors[0], 2), **kwargs
                    ),
                )

        elif ufunc in (
            np.divide,
            np.true_divide,
            np.floor_divide,
            np.remainder,
            np.mod,
            np.fmod,
            np.divmod,
        ):
            obj.errors = np.sqrt(
                np.add(
                    np.power(errors[0] / inp_cast[1], 2),
                    np.power(
                        inp_cast[0] / np.power(inp_cast[1], 2) * errors[1], 2
                    ),
                    **kwargs
                ),
            )

        elif ufunc in (np.power, np.float_power):
            obj.errors = np.sqrt(
                np.add(
                    np.power(
                        inp_cast[1]
                        * inp_cast[0] ** (inp_cast[1] - 1)
                        * errors[0],
                        2,
                    ),
                    np.power(
                        np.exp(inp_cast[1]) * np.log(inp_cast[0]) * errors[1],
                        2,
                    ),
                    **kwargs
                ),
            )
        elif ufunc == np.exp:
            obj.errors = np.sqrt(
                np.power(np.exp(inp_cast[0]) * errors[0], 2, **kwargs),
            )
        elif ufunc == np.log:
            obj.errors = np.sqrt(
                np.power(1 / inp_cast[0] * errors[0], 2, **kwargs)
            )
        elif ufunc == np.sqrt:
            obj.errors = np.sqrt(
                np.power(
                    1 / (2 * np.sqrt(inp_cast[0])) * errors[0], 2, **kwargs
                )
            )
        elif ufunc == np.square:
            obj.errors = np.sqrt(
                np.power(2 * inp_cast[0] * errors[0], 2, **kwargs)
            )
        elif ufunc == np.cbrt:
            obj.errors = np.sqrt(
                np.power(
                    1 / (3 * inp_cast[0] ** (2 / 3)) * errors[0], 2, **kwargs
                )
            )
        else:
            obj.errors = np.array(errors)

        return obj

    def __getitem__(self, key):
        arr = Sample(np.asarray(self)[key])
        arr.__dict__.update(self.__dict__)
        if np.asarray(self.errors).shape == self.shape:
            arr.errors = np.asarray(self.errors)[key]

        q = np.asarray(self.q)
        obs = getattr(self, self.observable)
        obs = np.asarray(obs)

        if isinstance(key, (int, slice)):
            if self.q.size in arr.shape:
                setattr(arr, self.observable, obs[key])
            if obs.size in arr.shape:
                arr.q = self.q[key]
        else:
            if len(key) == 1:
                if self.shape[0] == obs.size:
                    setattr(arr, self.observable, obs[key[0]])
                if self.shape[0] == q.size:
                    arr.q = self.q[key[0]]
            if len(key) == 2:
                if self.ndim == 1:
                    if self.shape[0] == obs.size:
                        setattr(arr, self.observable, obs[key])
                    if self.shape[0] == q.size:
                        arr.q = self.q[key]
                if self.ndim == 2:
                    if self.shape[0] == obs.size:
                        setattr(arr, self.observable, obs[key[0]])
                        arr.q = q[key[1]] if q.size > 1 else q
                    if self.shape[0] == q.size:
                        setattr(
                            arr,
                            self.observable,
                            obs[key[1]] if obs.size > 1 else obs,
                        )
                        arr.q = self.q[key[0]]

        return arr

    @property
    def T(self):
        arr = self.transpose()
        arr.errors = self.errors.transpose()

        return arr

    def bin(self, bin_size, *metadata, axis=0):
        """Bin data with the given bin size along specified axis.

        Parameters
        ----------
        bin_size : int
            The size of the bin (in number of data points).
        metadata : strings
            List of metadata names that should be binned as well.
        axis : int, optional
            The axis over which the binning is to be performed.
            (default, 0)

        Returns
        -------
        out_arr : :py:class:`Sample`
            A binned instance of :py:class:`Sample` with the same
            metadata except for **errors**, which are binned as well
            and possibly other user provided metadata names.

        """
        axis_size = self.shape[axis]
        nbr_iter = int(axis_size / bin_size)

        new_arr = []
        new_err = []
        for idx in range(nbr_iter):
            arr = self.take(
                np.arange(bin_size * idx, bin_size * idx + bin_size), axis
            ).mean(axis)
            err = self.errors.take(
                np.arange(bin_size * idx, bin_size * idx + bin_size), axis
            ).mean(axis)
            new_arr.append(arr)
            new_err.append(err)

        new_arr = np.array(new_arr)
        new_err = np.array(new_err)
        if axis == 1:
            new_arr = new_arr.T
            new_err = new_err.T

        new_meta = {}
        for meta_name in metadata:
            tmp_meta = []
            meta_axis = (
                axis if getattr(self, meta_name).ndim == self.ndim else -1
            )
            for idx in range(nbr_iter):
                tmp = (
                    getattr(self, meta_name)
                    .take(
                        np.arange(bin_size * idx, bin_size * idx + bin_size),
                        meta_axis,
                    )
                    .mean(meta_axis)
                )
                tmp_meta.append(tmp)
            new_meta[meta_name] = np.array(tmp_meta)

        out_arr = Sample(new_arr)
        out_arr.__dict__.update(self.__dict__)
        out_arr.__dict__.update(new_meta)
        out_arr.errors = np.array(new_err)

        return out_arr

    def sliding_average(self, window_size, *metadata, axis=0):
        """Performs a sliding average of data and errors along given axis.

        Parameters
        ----------
        window_size : int
            Size of the window for the sliding average.
        metadata : strings
            List of metadata names that should be binned as well.
        axis : int, optional
            The axis over which the average is to be performed.
            (default, 0)

        Returns
        -------
        out_arr : :py:class:`Sample`
            An averaged instance of :py:class:`Sample` with the same
            metadata except for **errors**, which are processed as well.

        """
        axis_size = self.shape[axis]
        last_idx = int(axis_size - window_size)

        new_arr = []
        new_err = []
        for idx in range(last_idx):
            arr = self.take(np.arange(idx, idx + window_size), axis).mean(axis)
            err = self.errors.take(
                np.arange(idx, idx + window_size), axis
            ).mean(axis)
            new_arr.append(arr)
            new_err.append(err)

        new_arr = np.array(new_arr)
        new_err = np.array(new_err)
        if axis == 1:
            new_arr = new_arr.T
            new_err = new_err.T

        new_meta = {}
        for meta_name in metadata:
            tmp_meta = []
            meta_axis = (
                axis if getattr(self, meta_name).ndim == self.ndim else -1
            )
            for idx in range(last_idx):
                tmp = (
                    getattr(self, meta_name)
                    .take(np.arange(idx, idx + window_size), meta_axis)
                    .mean(meta_axis)
                )
                tmp_meta.append(tmp)
            new_meta[meta_name] = np.array(tmp_meta)

        out_arr = Sample(new_arr)
        out_arr.__dict__.update(self.__dict__)
        out_arr.__dict__.update(new_meta)
        out_arr.errors = np.array(new_err)

        return out_arr

    def get_q_range(self, qmin, qmax):
        """Helper function to select a specific momentum transfer range.

        The function assumes that q values correspond to the last
        dimension of the data set.

        Parameters
        ----------
        qmin : int
            The minimum value for the momentum transfer q range.
        qmax : int
            The maximum value for the momentum transfer q range.

        Returns
        -------
        out : :py:class:`Sample`
            A new instance of the class with the selected q range.

        """
        if isinstance(self.q, int):
            print("No q values are available for this sample.\n")
            return

        min_idx = np.argmin((self.q - qmin) ** 2)
        max_idx = np.argmin((self.q - qmax) ** 2)

        out = self[..., min_idx:max_idx]

        return out

    def get_time_range(self, tmin, tmax):
        """Helper function to select a specific time range.

        The function assumes that time values correspond to the first
        dimension of the data set.

        Parameters
        ----------
        tmin : int
            The minimum value for time.
        tmax : int
            The maximum value for time.

        Returns
        -------
        out : :py:class:`Sample`
            A new instance of the class with the selected time range.

        """
        if isinstance(self.time, int):
            print("No time values are available for this sample.\n")
            return

        min_idx = np.argmin((self.time - tmin) ** 2)
        max_idx = np.argmin((self.time - tmax) ** 2)

        out = self[min_idx:max_idx]

        return out

    def plot(
        self,
        plot_type="standard",
        axis=0,
        xlabel=None,
        ylabel="I(q)",
        new_fig=False,
        max_lines=10,
        colormap="jet",
    ):
        """Helper function for quick plotting.

        Parameters
        ----------
        plot_type : str
            Type of plot to be generated (for q on x-axis).
            Possible options are:
                - **'standard'**, I(q) vs. q
                - **'q2'**, I(q) vs. q ** 2
                - **'log'**, log[I(q)] vs. q
                - **'guinier'**, log[I(q)] vs. q ** 2
                - **'kratky'**, q**2 * I(q) vs. q
        axis : int
            The axis along which to plot the data.
            If *xlabel* is None, then for 1D data, 0 is assumed to
            be q values, and for 2D data, 0 is assumed to correspond
            to time and 1 to q values.
        xlabel : str
            The label for the x-axis.
            (default None)
        ylabel : str
            The label for the y-axis.
            (default 'log(I(q))')
        new_fig : bool
            If true, create a new figure instead of plotting on the existing
            one.
        max_lines : int
            For 2D data, maximum number of lines to be plotted.
        colormap : str
            The colormap to be used for 2D data.

        """
        if new_fig or len(plt.get_fignums()) == 0:
            fig = plt.figure(figsize=(9, 6))
            if self.ndim == 1:
                ax = [fig.subplots(1, 1)]
            else:
                ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": (15, 1)})
        else:
            fig = plt.gcf()
            ax = fig.axes

        x = self.q if self.q.size == self.shape[axis] else self.time
        if xlabel is None:
            xlabel = (
                "q [$\\rm nm^{-1}$]"
                if x.size == self.q.size
                else self.observable
            )

        if plot_type in ["guinier", "q2"] and self.q.size == x.size:
            x = x ** 2
            xlabel = "$\\rm q^2$ [$\\rm nm^{-2}$]"

        if plot_type in ["log", "guinier"]:
            y = np.log(self)
            err = y.errors
            ylabel = "log[I(q)]"
        elif plot_type == "kratky":
            y = self.q ** 2 * self
            err = y.errors
            ylabel = "$\\rm q^2 I(q)$"
        else:
            y = self
            err = y.errors

        if self.ndim == 1:
            ax[0].plot(x, y, label=self.name)
            ax[0].fill_between(x, y - err, y + err, alpha=0.2)
        else:
            cmap = get_cmap(colormap)
            y = y.T if axis == 0 else y
            err = y.errors
            for idx, line in enumerate(
                y[:: int(np.ceil(y.shape[0] / max_lines))]
            ):
                ax[0].plot(x, line, color=cmap(idx / max_lines))
                ax[0].fill_between(
                    x,
                    line - err[idx],
                    line + err[idx],
                    color=cmap(idx / max_lines),
                    alpha=0.2,
                )

            cb_x = self.time if self.q.size == x.size else self.q
            norm = Normalize(cb_x[0], cb_x[-1])
            cb_label = (
                "q [$\\rm nm^{-1}$]"
                if xlabel == self.observable
                else self.observable
            )
            ColorbarBase(ax[1], cmap, norm, label=cb_label)

        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        ax[0].legend()
        plt.tight_layout()

        return fig, ax

    def range_selector(self, axis=0, sel_data=None, **kwargs):
        """Interactive plot to manually select a data range.

        Parameters
        ----------
        axis : int, optional
            The axis along which to plot the data.
            If *xlabel* is None, then for 1D data, 0 is assumed to
            be q values, and for 2D data, 0 is assumed to correspond
            to time and 1 to q values.
            (default, 0)
        sel_data : :py:class:`Sample`, optional
            An instance of :py:class:`Sample` class that will be used
            to perform the range selection.
            If None, the instance from which the method was called
            will be used.
            (default, None)
        kwargs : dict, keywords arguments
            Additional arguments to be passed to the :py:meth:`Sample.plot`
            method.

        """
        fig, ax = self.plot(axis=axis, new_fig=True, **kwargs)

        plt.title(
            "use left mouse button to select a data range\n"
            "use ctrl + left mouse button to select a buffer range"
        )

        if sel_data is None:
            sel_data = self

        return RangeSelector(fig, sel_data)

    def write_csv(self, filename, header=None, comments="# ", footer=None):
        """Write the data in text format.

        By default, the text contains three columns, the first one
        corresponding to q values, the second to I(q) and the third
        to the associated errors.

        If sample data are 2D, the data will be averaged over the first
        dimension before writing the file.

        Parameters
        ----------
        filename : str
            The name of the file to be written.
        header : str, optional
            A string defining the header of the file.
        comments : str, optional
            The string to be prepend to header lines to indicate a comment.
        footer : str, optional
            A string defining the footer of the file.

        """
        out = np.zeros((self.q.size, 3))

        if header is None:
            header = (
                "Name: %s\n"
                "Beamline: %s\n"
                "Detector: %s\n"
                "Wavelength: %s\n"
                "Flow rate: %s\n"
                "Temperature: %s\n"
                "Pressure: %s\n"
                "Concentration: %s\n"
                "Buffer: %s\n"
                "\n"
                "Following %i data points\n"
                "q [angs^-1]\t\tI(q)\t\terrors"
                % (
                    self.name,
                    self.beamline,
                    self.detectors,
                    self.wavelength,
                    self.flow_rate,
                    self.temperature,
                    self.pressure,
                    self.concentration,
                    self.buffer,
                    out.shape[0],
                )
            )

        if footer is None:
            footer = ""

        out[:, 0] = self.q
        out[:, 1] = self if self.ndim == 1 else self.mean(0)
        out[:, 2] = self.errors if self.ndim == 1 else self.mean(0).errors

        np.savetxt(
            filename, out, header=header, comments=comments, footer=footer
        )
