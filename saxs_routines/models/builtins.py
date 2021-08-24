"""This module provides several built-in models for incoherent
neutron scattering data fitting.

These functions generate a :py:class:`saxs_routines.models.model.Model`
class instance.

"""
import numpy as np

from saxs_routines.models.params import Parameters
from saxs_routines.models.model import Model, Component


# -------------------------------------------------------
# Built-in models
# -------------------------------------------------------
def model_linear_rg(name="linear_rg", **kwargs):
    """A linear model to extract the radius of gyration.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        I0={"value": 10.0, "bounds": (0.0, np.inf)},
        rg={"value": 2.0, "bounds": (0.0, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component("linear", lambda x, rg, I0: np.log(I0) - (x * rg) ** 2 / 3)
    )

    return m
