"""
Handle data associated with a sample.

"""


import numpy as np


class Sample(np.ndarray):
    """Handle the measured data along with metadata.

    This class is a subclass of the numpy.ndarray class with additional
    methods and attributes that are specific to small-angle
    X-ray scattering experiments on biological samples.

    It can handle various operations such as addition and subtraction
    of sample data or numpy array, scaling by a scalar or an array,
    binning, sliding average or data cleaning.

    Parameters
    ----------
    input_arr : np.ndarray, list, tuple or scalar
        Input array corresponding to sample scattering data.
    kwargs : dict (optional)
        Additional keyword arguments either for :py:meth:`np.asarray`
        or for sample metadata. The metadata are:
            - errors, the errors associated with scattering data.
            - time, the experimental time.
            - elution_volume, if available, the elution volume of the HPLC.
            - I0, the incoming beam intensity.
            - wavelength, the wavelength of the X-ray beam.
            - name, the name for the sample.
            - temperature, the temperature(s) used experimentally.
            - concentration, the concentration of the sample.
            - pressure, the pressure used experimentally.
            - buffer, a description of the buffer used experimentally.
            - q, the values for the momentum transfer q.
            - detector, the detector used.
            - beamline, the name of the beamline used.
            - flow_rate, the flow rate inside the capillary.

    Examples
    --------
    A sample can be created using the following:

    >>> s1 = Sample(my_data, errors=my_errors, name='protein1', q=q_values)

    where *my_data*, *my_errors* and *q_values* are numpy arrays.
    A buffer subtraction can be performed using:

    >>> s1 = s1 - buffer1

    where *buffer1* is another instance of :py:class:`Sample`. The error
    propagation is automatically performed and the other attributes are taken
    from the first operand (here s1).
    Other operations such as scaling can be performed using:

    >>> s1 = 0.8 * s1

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

        obj.__dict__.update(
            {
                key: val
                for key, val in kwargs.items()
                if key not in ("dtype", "order")
            }
        )

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.errors = getattr(obj, "errors", 0)
        self.time = getattr(obj, "time", 0)
        self.elution_volume = getattr(obj, "elution_volume", 0)
        self.I0 = getattr(obj, "I0", 0)
        self.wavelength = getattr(obj, "wavelength", 0)
        self.name = getattr(obj, "name", 0)
        self.temperature = getattr(obj, "temperature", 0)
        self.concentration = getattr(obj, "concentration", 0)
        self.pressure = getattr(obj, "pressure", 0)
        self.buffer = getattr(obj, "buffer", 0)
        self.q = getattr(obj, "q", 0)
        self.detectors = getattr(obj, "detectors", 0)
        self.beamline = getattr(obj, "beamline", 0)
        self.flow_rate = getattr(obj, "flow_rate", 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inp_cast = []
        for inp in inputs:
            if isinstance(inp, Sample):
                inp_cast.append(inp.view(np.ndarray))
            else:
                inp_cast.append(inp)

        obj = super().__array_ufunc__(ufunc, method, *inp_cast, **kwargs)
        if obj is NotImplemented:
            return NotImplemented

        obj = self._process_attributes(obj, ufunc, *inputs)

        return obj

    def _process_attributes(self, obj, ufunc, *inputs, **kwargs):
        print(inputs)
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
            obj.errors = np.sqrt(
                np.add(np.power(errors[0], 2), np.power(errors[1], 2)),
                **kwargs
            )
        elif ufunc == np.multiply:
            obj.errors = np.sqrt(
                np.add(
                    np.power(inp_cast[1] * errors[0], 2),
                    np.power(inp_cast[0] * errors[1], 2),
                ),
                **kwargs
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
                ),
                **kwargs
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
                ),
                **kwargs
            )
        elif ufunc == np.exp:
            obj.errors = np.sqrt(
                np.power(np.exp(inp_cast[0]) * errors[0], 2), **kwargs
            )
        elif ufunc == np.log:
            obj.errors = np.sqrt(
                np.power(1 / inp_cast[0] * errors[0], 2), **kwargs
            )
        elif ufunc == np.sqrt:
            obj.errors = np.sqrt(
                np.power(1 / (2 * np.sqrt(inp_cast[0])) * errors[0], 2),
                **kwargs
            )
        elif ufunc == np.square:
            obj.errors = np.sqrt(
                np.power(2 * inp_cast[0] * errors[0], 2), **kwargs
            )
        elif ufunc == np.cbrt:
            obj.errors = np.sqrt(
                np.power(1 / (3 * inp_cast[0] ** (2 / 3)) * errors[0], 2),
                **kwargs
            )
        else:
            print(
                "\nWarning!\n"
                "The universal function (%s) used is not handle by the "
                "'Sample' class.\nError propagation cannot be done "
                "automatically." % ufunc
            )

        return obj

    def bin(self, bin_size, axis=0):
        """Bin data with the given bin size along specified axis.

        Parameters
        ----------
        bin_size : int
            The size of the bin (in number of data points).
        axis : int, optional
            The axis over which the binning is to be performed.
            (default, 0)

        """
        pass
