#!/usr/bin/env python
# -*- coding: utf-8 -*-
# stats.py
"""Statistical calculations for the DVH Analytics"""
# Copyright (c) 2020 Dan Cutright
# Copyright (c) 2020 Arka Roy
#
# https://github.com/cutright/DVHA-Stats


from os.path import isfile, splitext
import numpy as np
from scipy.stats import beta
from src.utilities import dict_to_array, csv_to_dict
from scipy import stats as scipy_stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from regressors import stats as regressors_stats


class DVHAStats:
    def __init__(self, data, var_names=None):
        """Class used to calculated various statistics

        Parameters
        ----------
        data : numpy.array, dict, str
            Input data (2-D) with N rows of observations and
            p columns of variables.  The CSV file must have a header row
            for column names.
        var_names : list of str, optional
            If data is a numpy array, optionally provide the column names.

        """
        if isinstance(data, np.ndarray):
            self.data = data
            self.var_names = var_names
        elif isinstance(data, dict):
            data = dict_to_array(data)
            self.data = data["data"]
            self.var_names = data["var_names"]
        elif isfile(data) and splitext(data)[1] == ".csv":
            data = dict_to_array(csv_to_dict(data, dtype=float))
            self.data = data["data"]
            self.var_names = data["var_names"]
        else:
            msg = (
                "Invalid data provided - "
                "must be a numpy array, dict, or .csv file"
            )
            raise NotImplementedError(msg)

    @property
    def observations(self):
        """Number of observations"""
        return self.data.shape[0]

    @property
    def variable_count(self):
        """Number of variables"""
        return self.data.shape[1]

    def hotelling_t2(self, alpha=0.05):
        """Calculate Hotelling T^2

        Parameters
        ----------
        alpha : float
            The significance level used to calculate the
            upper control limit (UCL)

        Returns
        ----------
        dict
            The Hotelling T^2 values (Q), center line (CL),
            and upper control limit (UCL)

        """

        # Hotelling T^2 statistic
        Q = hotelling_t2(self.data)

        # Center Line
        x_cl = 0.5
        cl = hotelling_t2_control_limit(
            x_cl, self.observations, self.variable_count
        )

        # Upper Control Limit
        x_ucl = 1 - alpha / 2
        ucl = hotelling_t2_control_limit(
            x_ucl, self.observations, self.variable_count
        )

        # Out of Control indices
        ooc = [i for i, Qi in enumerate(Q) if Qi > ucl]

        return {"Q": Q, "CL": cl, "UCL": ucl, "OOC": ooc}

    def univariate_control_limits(self, std=3, ucl=None, lcl=None):
        """
        Calculate control limits for a standard univariate Control Chart

        Parameters
        ----------
        std : int, float, optional
            Number of standard deviations used to calculate if a y-value is
            out-of-control
        ucl : float, optional
            Over-ride the upper control limit
        lcl : float, optional
            Over-ride the lower control limit

        Returns
        ----------
        dict
            Output from get_control_limits stored in a dictionary with
            var_names as keys
        """
        kwargs = {"std": std, "ucl": ucl, "lcl": lcl}
        data = {}
        for i, key in enumerate(self.var_names):
            data[key] = get_control_limits(self.data[:, i], **kwargs)
        return data


class MultiVariableRegression:
    """Multi-variable regression using scikit-learn"""

    def __init__(self, X, y, saved_reg=None):
        """Initialization of a MultiVariableRegression

        Parameters
        ----------
        X : np.array
            Independent data
        y : np.array, list
            Dependent data
        saved_reg : MultiVariableRegression, optional
            Optionally provide a previously calculated regression
        """

        self.X = X
        self.y = np.array(y) if isinstance(y, list) else y

        self._do_fit(saved_reg=saved_reg)

    def _do_fit(self, saved_reg=None):
        """Perform linear regression

        Parameters
        ----------
        saved_reg : MultiVariableRegression, optional
            Optionally provide a previously calculated regression
        """
        if saved_reg is None:
            self.reg = linear_model.LinearRegression()
            self.ols = self.reg.fit(self.X, self.y)
        else:
            self.reg = saved_reg.reg
            self.ols = saved_reg.ols

        self.predictions = self.reg.predict(self.X)

        p_values = get_lin_reg_p_values(
            self.X, self.y, self.predictions, self.y_intercept, self.slope
        )
        self.p = p_values["p"]
        self.std_err = p_values["std_err"]
        self.t = p_values["t"]

        # ------------------------------------------
        # Calculate quantiles for a probability plot
        # ------------------------------------------
        norm_prob_plot = scipy_stats.probplot(
            self.residuals, dist="norm", fit=False, plot=None, rvalue=False
        )

        reg_prob = linear_model.LinearRegression()
        reg_prob.fit([[val] for val in norm_prob_plot[0]], norm_prob_plot[1])

        x_trend = [
            min(norm_prob_plot[0]),
            max(norm_prob_plot[0]),
        ]
        y_trend = np.add(
            np.multiply(x_trend, reg_prob.coef_),
            reg_prob.intercept_,
        )
        self.prob_plot = {
            "norm_prob_plot": norm_prob_plot,
            "y_intercept": reg_prob.intercept_,
            "slope": reg_prob.coef_,
            "trend": {"x": x_trend, "y": y_trend},
        }

    @property
    def y_intercept(self):
        """The y-intercept of the linear regression"""
        return self.reg.intercept_ if self.reg is not None else None

    @property
    def slope(self):
        """The slope of the linear regression"""
        return self.reg.coef_ if self.reg is not None else None

    @property
    def r_sq(self):
        """R^2 (coefficient of determination) regression score function."""
        return r2_score(self.y, self.predictions)

    @property
    def mse(self):
        """Mean squared error of the linear regression"""
        return mean_squared_error(self.y, self.predictions)

    @property
    def residuals(self):
        """Residuals of the prediction and sample data"""
        return np.subtract(self.y, self.predictions)

    @property
    def f_stat(self):
        """The F-statistic of the regression"""
        return regressors_stats.f_stat(self.ols, self.X, self.y)

    @property
    def df_error(self):
        """Error degrees of freedom"""
        return len(self.X[:, 0]) - len(self.X[0, :]) - 1

    @property
    def df_model(self):
        """Model degrees of freedom"""
        return len(self.X[0, :])

    @property
    def f_p_value(self):
        return scipy_stats.f.cdf(self.f_stat, self.df_model, self.df_error)


def hotelling_t2(arr):
    """Calculate Hotelling T^2 from a numpy array

    Parameters
    ----------
    arr : np.array
        A numpy array with N rows (observations) and p columns (variables)

    Returns
    -------
    np.array
        A numpy array of Hotelling T^2 (1-D of length N)
    """
    Q = np.zeros(np.size(arr, 0))
    D_bar = np.mean(arr, axis=0)
    S = np.cov(arr.T)
    S_inv = np.linalg.inv(S)
    observations = np.size(arr, 0)
    for i in range(observations):
        spread = arr[i, :] - D_bar
        Q[i] = np.matmul(np.matmul(spread, S_inv), spread)
    return Q


def hotelling_t2_control_limit(x, observations, variables):
    """Calculate a Hotelling T^2 control limit using a beta distribution

    Parameters
    ----------
    x : float
        Value where the beta function is evaluated
    observations : int
        Number of observations in the sample
    variables : int
        Number of variables observed for each sample

    Returns
    -------
    float
        The control limit for a beta distribution using the provided parameters
    """

    N = observations
    a = variables / 2
    b = (N - variables - 1) / 2
    return ((N - 1) ** 2 / N) * beta.ppf(x, a, b)


def get_control_limits(y, std=3, ucl=None, lcl=None):
    """
    Calculate control limits for a standard univariate Control Chart

    Parameters
    ----------
    y : list, np.array
        Input data (1-D)
    std : int, float, optional
        Number of standard deviations used to calculate if a y-value is
        out-of-control.
    ucl : float, optional
        Over-ride the upper control limit
    lcl : float, optional
        Over-ride the lower control limit

    Returns
    ----------
    dict
        The center line (CL),  upper/lower control limits (LCL, UCL), and
        sigma (sigma) so control limits can be recalculated.
    """

    if isinstance(y, list):
        y = np.array(y)

    center_line = np.mean(y)
    avg_moving_range = np.mean(np.absolute(np.diff(y)))

    # since moving range is calculated based on 2 consecutive points
    scalar_d = 1.128

    sigma = avg_moving_range / scalar_d

    ucl = center_line + std * sigma if ucl is None else ucl
    lcl = center_line - std * sigma if lcl is None else lcl

    return {
        "CL": center_line,
        "UCL": ucl,
        "LCL": lcl,
        "sigma": sigma,
        "OOC": [i for i, yi in enumerate(y) if yi > ucl or yi < lcl],
        "OOC_high": [i for i, yi in enumerate(y) if yi > ucl],
        "OOC_low": [i for i, yi in enumerate(y) if yi < lcl],
    }


def get_lin_reg_p_values(X, y, predictions, y_intercept, slope):
    """
    Get p-values of a linear regression using sklearn
    based on https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

    Parameters
    ----------
    X : np.array
        Independent data
    y : np.array, list
        Dependent data
    predictions : np.array, list
        Predictions using the linear regression.
        (Output from linear_model.LinearRegression.predict)
    y_intercept : float
        The y-intercept of the linear regression
    slope : float
        The slope of the linear regression

    Returns
    ----------
    dict
        A dictionary of p-values (p), standard errors (std_err),
        and t-values (t)
    """

    if isinstance(y, list):
        y = np.array(y)

    if isinstance(predictions, list):
        predictions = np.array(predictions)

    newX = np.append(np.ones((len(X), 1)), X, axis=1)
    mse = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

    variance = mse * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    std_errs = np.sqrt(variance)

    params = np.append(y_intercept, slope)
    t_values = np.array(params) / std_errs

    p_value = [
        2 * (1 - scipy_stats.t.cdf(np.abs(i), (len(newX) - 1)))
        for i in t_values
    ]

    return {"p": p_value, "std_err": std_errs, "t": t_values}
