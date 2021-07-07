"""This module provides a routine to compute the radius of gyration.

"""


import numpy as np

from scipy.optimize import curve_fit

from saxs_routines.models.builtins import model_linear_rg


class AutoRg:
    """Computes a radius of gyration for the provided data.

    The methods ...

    Parameters
    ----------

    """

    def __init__(self, data, qmin=0.1, qmax=0.4):
        self.data = data
        self.qmin = qmin
        self.qmax = qmax

        self.rg = None
        self.rg_std = None
        self.I0 = None
        self.I0_std = None

    def run(self):
        """Run the algorithm to extract the radius of gyration.

        The result is stored in the *rg*, *I0*, *rg_std* and
        *I0_std* attributes.

        """
