"""
This module provides a template class to build models
that can be used to fit the data.

"""
import sys

from copy import deepcopy
from inspect import signature
import ast

import operator

from collections import OrderedDict

import numpy as np

from scipy.optimize import (
    curve_fit,
    minimize,
    basinhopping,
    differential_evolution,
    shgo,
)
from scipy.sparse.linalg import LinearOperator
from scipy.signal import fftconvolve
from scipy.special import wofz, spherical_jn, erf, erfc, gamma

from saxs_routines.models import Parameters

_PY_VERSION = sys.version_info
_PY_VERSION = float("%d.%d" % (_PY_VERSION.major, _PY_VERSION.minor))


class FindParamNames(ast.NodeTransformer):
    """Helper class to parse strings to evaluation for function
    arguments in :class:`Component`.

    Parameters
    ----------
    params : :class:`Parameters`
        An instance of Parameters from which the parameter
        names are to be found and substituted by the corresponding
        values.

    """

    def __init__(self, key, params):
        super().__init__()
        self.key = key
        self.params = params

    def visit_Name(self, node):
        """Name visitor."""
        if node.id in self.params.keys():
            if _PY_VERSION <= 3.6:
                sliceVal = ast.Index(value=ast.Str(s=node.id))
            else:
                sliceVal = ast.Index(ast.Constant(s=node.id))

            res = ast.Attribute(
                value=ast.Subscript(
                    value=ast.Name(id="params", ctx=node.ctx),
                    slice=sliceVal,
                    ctx=node.ctx,
                ),
                attr="value",
                ctx=node.ctx,
            )
            return res
        return node


class Model:
    """Model class to be used for fitting.

    The model is structured in components that can be added
    together, each component consisting of a name, a callable
    function and a dictionary of parameters. The parameters
    of two different components can have the same name such
    that they can be shared by several components just like
    for the switching diffusive state model.

    Also, the components are separated in two classes, namely
    *eisfComponents* and *qisfComponents*, in order to
    provide the possibility to separately extract the elastic
    and quasi-elastic parts for analysis and plotting.

    Parameters
    ----------
    params : :class:`Parameters` instance
        Parameters to be used with the model
    name : str, optional
        A name for the model.
    convolutions : dict of dict
        Dictionary that defines the mapping '(function1, function2)'
        to 'convolutionFunction(function1, function2)'. Analytic
        convolutions or user defined operators can be defined
        this way.
    on_undef_conv : {'raise', 'numeric'}
        Defines the behavior of the class on missing convolution function
        in the 'convolutions' attribute. The option 'raise' leads to a
        `KeyError` and the option 'numeric' to a numerical convolution.

    """

    _opMap = {
        "+": operator.add,
        "-": operator.sub,
        "/": operator.truediv,
        "*": operator.mul,
    }

    def __init__(
        self, params, name="Model", convolutions=None, on_undef_conv="numeric"
    ):
        self.name = name
        self.params = deepcopy(params) if params is not None else {}
        self._components = OrderedDict()
        self._operators = []

        self.convolutions = {}

        if convolutions is not None:
            for key, val in convolutions.items():
                self.convolutions[key].update(val)

        self._optParams = None
        self._fitResult = None
        self._bic = None
        self._userkws = {}

        self._on_undef_conv = on_undef_conv

    @property
    def components(self):
        """Return the model components."""
        return self._components

    def addComponent(self, comp, op="+"):
        """Add a component to the model.

        Parameters
        ----------
        comp : :class:`Component`
            An instance of `Component` to be added to the model.
        op : {"+", "-", "*", "/"}, optional
            Operator to be used to combine the new component with the others.
            If this is the first component, the operator is ignored.
            (default "+")

        """
        if len(self._components.keys()) > 0:
            self._operators.append(self._opMap[op])
        self._components[comp.name] = comp

    @property
    def on_undef_conv(self):
        """Return the class behavior on undefined convolution."""
        return self._on_undef_conv

    @on_undef_conv.setter
    def on_undef_conv(self, val):
        """Setter for 'on_undef_conv'"""
        if val not in ["numeric", "raise"]:
            raise ValueError(
                "The attribute 'on_undef_conv' can only be "
                "'numeric' or 'raise'."
            )
        self._on_undef_conv = val

    # --------------------------------------------------
    # fitting
    # --------------------------------------------------
    @property
    def optParams(self):
        """Return the result of the fit."""
        if self._optParams is not None:
            return self._optParams
        raise ValueError(
            "No optimal parameters found for this model "
            "(Model named '{name}' at {address}).\n"
            "Please use 'fit' method to optimize the parameters".format(
                name=self.name, address=hex(id(self))
            )
        )

    @optParams.setter
    def optParams(self, params):
        """Setter for the optimized parameters."""
        if not isinstance(params, Parameters):
            raise ValueError(
                "The 'optParams' attribute should contain an instance "
                "of a 'Parameters' object."
            )
        self._optParams = params

    @property
    def bic(self):
        """Return the bayesian information criterion (BIC)."""
        return self._bic

    @property
    def userkws(self):
        """Return the keywords used for the fit."""
        return self._userkws

    @property
    def fitResult(self):
        """Return the full result of the fit."""
        return self._fitResult

    def fit(
        self,
        x,
        data=None,
        weights=None,
        fit_method="curve_fit",
        fit_kws=None,
        params=None,
        **kwargs
    ):
        """Fit the experimental data using the provided arguments.

        Parameters
        ----------
        x : np.ndarray
            Values for the indenpendent variable.
        data : np.ndarray
            Experimental data to be fitted.
        weights : np.ndarray, optional
            Weights associated with the experimental data (the
            experimental errors).
        fit_method : str, optional
            The method to be used for fitting.
            Currently available methods are (from Scipy):
            - "curve_fit"
            - "basinhopping"
            - "differential_evolution"
            - "shgo"
            - "minimize"
        fit_kws : dict, optional
            Additional keywords to be passed to the fit method.
        params : :class:`Parameters` class instance, optional
            Parameters to be used (default None, will use the parameters
            associated with the model).
        kwargs : dict, optional
            Additional keywords arguments to give for the evaluation
            of the model. Can override parameters too.

        Returns
        -------
        A copy of the fitted model instance.

        """
        # process 'x' array and match the shape of data
        params, bounds = self.params.paramList

        if fit_kws is None:
            fit_kws = {}

        norm = 1 / np.sum(data ** 2 / weights ** 2)
        func = lambda p: norm * np.sum(
            (self.eval(x, p, **kwargs) - data) ** 2 / weights ** 2
        )

        if fit_method == "curve_fit":
            func = lambda x, *p: self.eval(x, p, **kwargs).flatten()
            bounds = ([val[0] for val in bounds], [val[1] for val in bounds])
            fit = curve_fit(
                func,
                x,
                data.flatten(),
                params,
                weights.flatten(),
                bounds=bounds,
                **fit_kws
            )
            nbrParams = len(fit[0])
            self.optParams = self.params.listToParams(
                fit[0], np.sqrt(np.diag(fit[1]))
            )

        if fit_method == "basinhopping":
            if "minimizer_kwargs" in fit_kws.keys():
                fit_kws["minimizer_kwargs"].update(bounds=bounds)
            else:
                fit_kws["minimizer_kwargs"] = {"bounds": bounds}
            fit = basinhopping(func, params, **fit_kws)

            paramErr = fit.lowest_optimization_result.hess_inv
            if isinstance(paramErr, LinearOperator):
                paramErr = paramErr.todense()
            nbrParams = len(fit.x)
            self.optParams = self.params.listToParams(
                fit.x, np.sqrt(np.diag(paramErr))
            )

        if fit_method == "shgo":
            fit = shgo(func, bounds, **fit_kws)

            nbrParams = len(fit.x)
            self.optParams = self.params.listToParams(fit.x)

        if fit_method == "minimize":
            fit = minimize(func, params, bounds=bounds, **fit_kws)

            paramErr = fit.hess_inv.todense()
            nbrParams = len(fit.x)
            self.optParams = self.params.listToParams(
                fit.x, np.sqrt(np.diag(paramErr))
            )

        if fit_method == "differential_evolution":
            maxFloat = np.finfo("float64").max
            minFloat = np.finfo("float64").min
            for boundsIdx, boundsTuple in enumerate(bounds):
                pMin, pMax = boundsTuple
                if not np.isfinite(pMin):
                    pMin = minFloat
                if not np.isfinite(pMax):
                    pMax = maxFloat
                bounds[boundsIdx] = (pMin, pMax)

            fit = differential_evolution(func, bounds=bounds, **fit_kws)

            nbrParams = len(fit.x)
            self.optParams = self.params.listToParams(fit.x)

        # computes the bayesian information criterion
        self._bic = nbrParams * np.log(x.size) + 2 * np.log(
            np.exp(
                -norm
                * (
                    (self.eval(x, self.optParams, **kwargs) - data) ** 2
                    / weights ** 2
                ).sum()
            )
        )

        self._userkws["x"] = x
        self._userkws["data"] = data
        self._userkws["weights"] = weights
        self._userkws["params"] = params
        self._userkws.update(**kwargs)
        self._fitResult = fit

        return self.copy()

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------
    def eval(self, x, params=None, convolve=None, **kwargs):
        """Perform the assembly of the components and call
         the provided functions with their parameters to
         compute the model.

        Parameters
        ----------
        x : np.ndarray
            Values for the x-axis variable
        params : list, np.array, optional
            Parameters to be passed to the components.
            Will override existing parameters in `self.params`.
        convolve : :class:`Model`
            Another model to be convolved with this one.
        kwargs:
            Additional keyword arguments to be passed to the components.
            Can override params too.

        Returns
        -------
        If `returnComponents` is False:
            The computed model in an array, the dimensions of which depend
            on `x` and `params` attributes and the function called.
        else:
            A dictionary with key being the component names and the values
            are the evaluated components.

        """
        names, res = self._getComponents(x, params, convolve, **kwargs)
        for idx, val in enumerate(res[1:]):
            res[0] = self._operators[idx](res[0], val)

        return res[0]

    def eval_components(self, x, params=None, convolve=None, **kwargs):
        """Alias for `eval` with 'returnComponents' set to True.

        Perform the computation of the components
        with the given x-axis values, parameters and convolutions.

        Returns
        -------
        A dictionary with key being the component names and the values
        are the evaluated components.

        """
        names, comps = self._getComponents(x, params, convolve, **kwargs)
        return {name: comps[idx] for idx, name in enumerate(names)}

    def copy(self):
        """Return a copy of the model."""
        m = Model(
            self.params, self.name, self.convolutions, self._on_undef_conv
        )
        for key, val in self._components.items():
            m._components[key] = deepcopy(val)
        m._operators = deepcopy(self._operators)
        m._optParams = deepcopy(self._optParams)
        m._userkws = deepcopy(self._userkws)
        m._fitResult = deepcopy(self._fitResult)
        m._bic = deepcopy(self._bic)

        return m

    def _getComponents(self, x, params, convolve, **kwargs):
        """Return the evaluated components."""
        names = []
        comps = []

        # if no parameters override, use the ones from the class instance
        if params is None:
            params = deepcopy(self.params)
        else:
            if isinstance(params, dict):
                params.update(**params)
            else:  # assumes a list or numpy array
                params = self.params.listToParams(params)

        # gets the output arrays for each component and sum
        for key, comp in self.components.items():
            names.append(comp.name)
            if isinstance(comp, Component):
                if convolve is None or comp.skip_convolve:
                    comps.append(comp.eval(x, params, **kwargs))
                else:
                    comps.append(
                        self._convolve(x, params, comp, convolve, **kwargs)
                    )
            if isinstance(comp, Model):
                model_kws = {}
                try:
                    compParams = comp.optParams
                    model_kws = comp.userkws
                    if "x" in model_kws:
                        model_kws.pop("x")
                    if "params" in model_kws:
                        model_kws.pop("params")
                except ValueError:
                    print(
                        "No fitted parameters found for the sub-model: "
                        "{name} at {address}.\n"
                        "Using initial parameters instead.".format(
                            name=comp.name, address=hex(id(comp))
                        )
                    )
                    compParams = comp.params
                comps.append(comp.eval(x, compParams, convolve, **model_kws))

        return names, comps

    def _convolve(self, x, params, comp, convolve, **kwargs):
        """This method allows to identify the type of function
        given in the two components of the convolution.

        If these function are referred to in the `convolutions`
        attribute of the class, an analytical convolution will
        be performed using the corresponding convolution function.
        Otherwise, a numerical convolution is performed.

        Parameters
        ----------
        x : np.ndarray
            Array for values corresponding to the x-axis
        params : list, np.array, optional
            Parameters to be passed to the components.
        comp : tuple(function, parameters, [convolutions])
            Component of the current :class:`Model` instance
            to be convolved with the components of the other
            model.
        convolve : :class:`Model`
            Another :class:`Model` instance that is used for
            convolution.
        kwargs:
            Additional keyword arguments to be passed to the components.

        """
        try:
            convParams = convolve.optParams
        except ValueError:
            print(
                "No fitted parameters found for the model to convolve: "
                "{name} at {address}.\n"
                "Using initial parameters instead.".format(
                    name=comp.name, address=hex(id(comp))
                )
            )
            convParams = convolve.params

        res = []
        if comp.func.__name__ in self.convolutions.keys():
            convDict = self.convolutions[comp.func.__name__]
            for key, val in convolve.components.items():
                # if no convolution defined, go numerical or raise KeyError
                if val.func.__name__ not in convDict.keys():
                    if self._on_undef_conv == "numeric":
                        shift = int(round(x.size / 2)) - np.argmin(np.abs(x))
                        res.append(
                            fftconvolve(
                                np.roll(
                                    comp.eval(x, params, **kwargs), shift, -1
                                ),
                                val.eval(x, convParams, **kwargs),
                                mode="same",
                                axes=-1,
                            )
                        )
                    else:
                        raise KeyError(
                            "The convolution function between "
                            "{func1} and {func2} is not defined "
                            "in the 'convolutions' "
                            "attribute.".format(
                                func1=comp.func.__name__,
                                func2=val.func.__name__,
                            )
                        )
                else:
                    convFunc = convDict[val.func.__name__]
                    res.append(
                        convFunc(x, comp, val, params, convParams, **kwargs)
                    )
        else:
            for key, val in convolve.components.items():
                if self._on_undef_conv == "numeric":
                    shift = int(round(x.size / 2)) - np.argmin(np.abs(x))
                    res.append(
                        fftconvolve(
                            np.roll(comp.eval(x, params, **kwargs), shift, -1),
                            val.eval(x, convParams, **kwargs),
                            mode="same",
                            axes=-1,
                        )
                    )
                else:
                    raise KeyError(
                        "The convolution function between "
                        "{func1} and {func2} is not defined in "
                        "the 'convolutions' attribute.".format(
                            func1=comp.func.__name__, func2=val.func.__name__
                        )
                    )

        # apply the operators from 'model'
        for idx, val in enumerate(res[1:]):
            res[0] = convolve._operators[idx](res[0], val)

        return res[0]

    def __add__(self, other):
        """Addition operator between Model and Component."""
        if not isinstance(other, (Model, Component)):
            raise TypeError(
                "The Model class currently supports addition "
                "with a 'Component' or 'Model' instance only."
            )
        m = self.copy()
        m.addComponent(other, "+")

        return m

    def __sub__(self, other):
        """Subtraction operator between Model and Component."""
        if not isinstance(other, (Model, Component)):
            raise TypeError(
                "The Model class currently supports subtraction "
                "with a 'Component' or 'Model' instance only."
            )
        m = self.copy()
        m.addComponent(other, "-")

        return m

    def __truediv__(self, other):
        """Division operator between Model and Component."""
        if not isinstance(other, (Model, Component)):
            raise TypeError(
                "The Model class currently supports division "
                "with a 'Component' or 'Model' instance only."
            )
        m = self.copy()
        m.addComponent(other, "/")

        return m

    def __mul__(self, other):
        """Multplication operator between Model and Component."""
        if not isinstance(other, (Model, Component)):
            raise TypeError(
                "The Model class currently supports "
                "mulitplication with a 'Component' or 'Model' "
                "instance only."
            )
        m = self.copy()
        m.addComponent(other, "*")

        return m


class Component:
    """Component class to be used with the :class:`Model` class.

    Parameters
    ----------
    name : str
        Name for the component.
    func : callable
        The function to be used for this component.
    skip_convolve : bool
        If True, no convolution is performed for this model.
        It can be useful for background or normalization terms.
    funcArgs : dict of str, int, float or arrays
        Values to be passed to the function arguments.
        This is a dicitonary of argument names mapped to values.
        The values can be of different types:
            - **int, float or array**, the values are directly passed to
              the function.
            - **str**, the values are evaluated first. If any word in
              the string is present in the `Model.params` dictionary keys,
              the corresponding parameter value is substituted.

    Examples
    --------
    For a `Model` class that has the following key in its `params`
    attribute: ('amplitude', 'sigma'), the component for a
    Lorentzian, the width of which depends on a defined vector *q*,
    can be created using:

    >>> def lorentzian(x, scale, width):
    ...     return scale / np.pi * width / (x**2 + width**2)
    >>> myComp = Component(
    ...     'lor', lorentzian, scale='scale', width='width * q**2')

    If the Lorentzian width is constant, use:

    >>> myComp = Component('lor', lorentzian, scale='scale', width=5)

    Some math functions can be used as well (below the exponential):

    >>> myComp = Component('lor', lorentzian, scale='np.exp(-q**2 * msd)')

    """

    def __init__(self, name, func, skip_convolve=False, **funcArgs):
        self.name = name

        if not hasattr(func, "__call__"):
            raise AttributeError("Parameter 'func' should be a callable.")
        self.func = func
        self.skip_convolve = skip_convolve

        self.funcArgs = {}
        self._guessArgs()
        self.funcArgs.update(funcArgs)

    def eval(self, x, params, **kwargs):
        """Evaluate the components using the given parameters.

        Parameters
        ----------
        params : :class:`Parameters` instance
            Parameters to be passed to the component
        kwargs : dict
            Additional parameters to be passed to the function.
            Can override params.

        """
        args = self.processFuncArgs(params, **kwargs)
        return self.func(x, **args)

    def processFuncArgs(self, params, **kwargs):
        """Return the evaluated argument for the function using given
        parameters and keyword arguments.

        """
        params = deepcopy(params)
        for key, val in kwargs.items():
            if key in params.keys():
                params.update(**{key: val})
            else:
                params.set(key, value=val, fixed=True)

        args = {}
        for key, arg in self.funcArgs.items():
            if isinstance(arg, str):
                arg = ast.parse(arg, mode="eval")
                arg = ast.fix_missing_locations(
                    FindParamNames(key, params).visit(arg)
                )
                c = compile(arg, "<string>", "eval")
                args[key] = eval(c)
            else:
                args[key] = arg

        return args

    def _guessArgs(self):
        """Guess arguments from function signature."""
        sig = signature(self.func)
        for key, param in sig.parameters.items():
            if key != "x":
                self.funcArgs[key] = key
