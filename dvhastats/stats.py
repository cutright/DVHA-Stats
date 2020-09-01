#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# stats.py
"""Statistical calculations for DVH Analytics"""
#
# Copyright (c) 2020 Dan Cutright
# Copyright (c) 2020 Arka Roy
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


from os.path import isfile, splitext
import numpy as np
from scipy.stats import beta
from dvhastats.utilities import (
    dict_to_array,
    csv_to_dict,
    close_plot,
    moving_avg,
)
from dvhastats import plot
from scipy import stats as scipy_stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from regressors import stats as regressors_stats


class DVHAStats:
    def __init__(self, data, var_names=None, x_axis=None, avg_len=5):
        """Class used to calculated various statistics

        Parameters
        ----------
        data : numpy.array, dict, str
            Input data (2-D) with N rows of observations and
            p columns of variables.  The CSV file must have a header row
            for column names.
        var_names : list of str, optional
            If data is a numpy array, optionally provide the column names.
        x_axis : numpy.array, list, optional
            Specify x_axis for plotting purposes. Default is based on row
            number in data
        avg_len : int
            When plotting raw data, a trend line will be plotted using this
            value as an averaging length. If N < avg_len + 1 will not
            plot a trend line
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

        self.x_axis = x_axis

        self.box_cox_data = None

        self.plots = []

        self.avg_len = avg_len

    def get_data_by_var_name(self, var_name):
        """Get the single variable array based on var_name"""
        index = self.get_index_by_var_name(var_name)
        return self.data[:, index]

    def get_index_by_var_name(self, var_name):
        """Get the variable index by var_name"""
        if var_name in self.var_names:
            index = self.var_names.index(var_name)
        elif isinstance(var_name, int) and var_name in range(
            self.variable_count
        ):
            return var_name
        else:
            msg = "%s is not a valid var_name\n%s" % (
                var_name,
                ",".join(self.var_names),
            )
            raise AttributeError(msg)
        return index

    @property
    def observations(self):
        """Number of observations"""
        return self.data.shape[0]

    @property
    def variable_count(self):
        """Number of variables"""
        return self.data.shape[1]

    @property
    def pearson_r_matrix(self):
        """Get a Pearson-R correlation matrix and associated p-value matrix

        Returns
        ----------
        np.ndarray, np.ndarray
            A tuple of symmetric, pxp arrays are returned: PearsonR and its
            p-values.
        """
        return pearson_r_matrix(self.data)

    @property
    def normality(self):
        """Calculate the normality and associated p-values for each variable

        Returns
        ----------
        np.ndarray, np.ndarray
            A tuple of 1-D arrays are returned: Normality and its
            p-values.
        """
        norm, p = np.zeros(self.variable_count), np.zeros(self.variable_count)
        for i in range(self.variable_count):
            norm[i], p[i] = scipy_stats.normaltest(
                self.data[:, i], nan_policy="omit"
            )
        return norm, p

    def linear_reg(self, y, saved_reg=None):
        """Initialize a MultiVariableRegression class object

        Parameters
        ----------
        y : np.ndarray, list
            Dependent data based on DVHAStats.data
        saved_reg : MultiVariableRegression, optional
            If supplied, predicted values (y-hat) will be calculated with
            DVHAStats.data and the regression from saved_reg. This is useful
            if testing a regression model on new data.

        Returns
        ----------
        MultiVariableRegression
            A MultiVariableRegression class object.
        """
        return MultiVariableRegression(self.data, y, saved_reg)

    def univariate_control_charts(
        self,
        std=3,
        ucl_limit=None,
        lcl_limit=None,
        box_cox=False,
        box_cox_alpha=None,
        box_cox_lmbda=None,
    ):
        """
        Calculate control limits for a standard univariate Control Chart

        Parameters
        ----------
        std : int, float, optional
            Number of standard deviations used to calculate if a y-value is
            out-of-control
        ucl_limit : float, optional
            Limit the upper control limit to this value
        lcl_limit : float, optional
            Limit the lower control limit to this value
        box_cox : bool, optional
            Set to true to perform a Box-Cox transformation on data prior to
            calculating the control chart statistics
        box_cox_alpha : float, optional
            If alpha is not None, return the 100 * (1-alpha)% confidence
            interval for lmbda as the third output argument. Must be between
            0.0 and 1.0.
        box_cox_lmbda : float, optional
            If lmbda is not None, do the transformation for that value.
            If lmbda is None, find the lambda that maximizes the log-likelihood
            function and return it as the second output argument.

        Returns
        ----------
        dict
            ControlChartData class objects stored in a dictionary with
            var_names and indices as keys (can use var_name or index)
        """
        kwargs = {"std": std, "ucl_limit": ucl_limit, "lcl_limit": lcl_limit}
        data = {}
        if box_cox:
            if self.box_cox_data is None:
                self.box_cox(alpha=box_cox_alpha, lmbda=box_cox_lmbda)
            cc_data = self.box_cox_data
            plot_title = "Univariate Control Chart with Box-Cox Transformation"
        else:
            cc_data = self.data
            plot_title = None
        for i, key in enumerate(self.var_names):
            data[key] = ControlChartData(
                cc_data[:, i], var_name=key, plot_title=plot_title, **kwargs
            )
            data[i] = data[key]
        return data

    def hotelling_t2(
        self, alpha=0.05, box_cox=False, box_cox_alpha=None, box_cox_lmbda=None
    ):
        """
        Calculate control limits for a standard univariate Control Chart

        Parameters
        ----------
        alpha : float
            Significance level used to determine the upper control limit (ucl)
        box_cox : bool, optional
            Set to true to perform a Box-Cox transformation on data prior to
            calculating the control chart statistics
        box_cox_alpha : float, optional
            If alpha is not None, return the 100 * (1-alpha)% confidence
            interval for lmbda as the third output argument. Must be between
            0.0 and 1.0.
        box_cox_lmbda : float, optional
            If lmbda is not None, do the transformation for that value.
            If lmbda is None, find the lambda that maximizes the log-likelihood
            function and return it as the second output argument.

        Returns
        ----------
        HotellingT2
            HotellingT2 class object
        """

        if box_cox:
            if self.box_cox_data is None:
                self.box_cox(alpha=box_cox_alpha, lmbda=box_cox_lmbda)
            data = self.box_cox_data
            plot_title = (
                "Multivariate Control Chart with Box-Cox Transformation"
            )
        else:
            data = self.data
            plot_title = None
        return HotellingT2(data, alpha, plot_title=plot_title)

    def box_cox_by_index(self, index, alpha=None, lmbda=None):
        """

        Parameters
        ----------
        index : int, str
            The index corresponding to the variable data to have a box-cox
            transformation applied.  If index is a string, it will be assumed
            to be the var_name
        lmbda : None, scalar, optional
            If lmbda is not None, do the transformation for that value.
            If lmbda is None, find the lambda that maximizes the
            log-likelihood function and return it as the second output
            argument.
        alpha : None, float, optional
            If alpha is not None, return the 100 * (1-alpha)% confidence
            interval for lmbda as the third output argument. Must be between
            0.0 and 1.0.
        """
        if self.box_cox_data is None:
            self.box_cox_data = np.zeros_like(self.data)

        if isinstance(index, str):
            index = self.get_index_by_var_name(index)

        self.box_cox_data[:, index], _ = scipy_stats.boxcox(
            self.data[:, index], alpha=alpha, lmbda=lmbda
        )

        return self.box_cox_data[:, index]

    def box_cox(self, alpha=None, lmbda=None):
        """Apply box_cox_by_index for all data"""
        for i in range(self.variable_count):
            self.box_cox_by_index(i, alpha=alpha, lmbda=lmbda)

    def add_tend_line(self, var_name, plot_index):
        trend_x, trend_y = moving_avg(
            self.get_data_by_var_name(var_name), self.avg_len
        )
        self.plots[plot_index].add_line(
            trend_y, x=trend_x, line_color="black", line_width=0.75
        )

    def show(self, var_name):
        """Display a plot of var_name with matplotlib"""
        index = self.get_index_by_var_name(var_name)
        var_name = self.var_names[index]
        self.plots.append(
            plot.Plot(
                self.data[:, index],
                x=self.x_axis,
                xlabel="Observation",
                ylabel=var_name,
            )
        )
        self.add_tend_line(var_name, -1)
        return self.plots[-1].figure.number

    def close(self, figure_number):
        """Close a plot by figure_number"""
        close_plot(figure_number, self.plots)


def get_lin_reg_p_values(X, y, predictions, y_intercept, slope):
    """
    Get p-values of a linear regression using sklearn
    based on https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

    Parameters
    ----------
    X : np.ndarray
        Independent data
    y : np.ndarray, list
        Dependent data
    predictions : np.ndarray, list
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

    return p_value, std_errs, t_values


class MultiVariableRegression:
    """Multi-variable regression using scikit-learn"""

    def __init__(self, X, y, saved_reg=None):
        """Initialization of a MultiVariableRegression

        Parameters
        ----------
        X : np.ndarray
            Independent data
        y : np.ndarray, list
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

        self.p, self.std_err, self.t = get_lin_reg_p_values(
            self.X, self.y, self.predictions, self.y_intercept, self.slope
        )

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


def pearson_r_matrix(X):
    """Calculate a correlation matrix of Pearson-R values

    Parameters
    ----------
    X : np.ndarray
        Input data (2-D) with N rows of observations and
        p columns of variables.

    Returns
    ----------
    np.ndarray
        A tuple of symmetric, pxp arrays are returned: PearsonR and its
        p-values.
    """

    r = np.ones([X.shape[1], X.shape[1]])  # Pearson R
    p = np.zeros([X.shape[1], X.shape[1]])  # p-value of Pearson R

    for x in range(X.shape[1]):
        for y in range(X.shape[1]):
            if x > y:

                # Pearson r requires that both sets of data be of the same
                # length. Remove index if NaN in either variable.
                valid_x = ~np.isnan(X[:, x])
                valid_y = ~np.isnan(X[:, y])
                include = np.full(len(X[:, x]), True)
                for i in range(len(valid_x)):
                    include[i] = valid_x[i] and valid_y[i]
                x_data = X[include, x]
                y_data = X[include, y]

                r[x, y], p[x, y] = scipy_stats.pearsonr(x_data, y_data)

                # These matrices are symmetric
                r[y, x] = r[x, y]
                p[y, x] = p[x, y]

    return r, p


class ControlChartData:
    def __init__(
        self,
        y,
        std=3,
        ucl_limit=None,
        lcl_limit=None,
        var_name=None,
        x=None,
        plot_title=None,
    ):
        """
        Calculate control limits for a standard univariate Control Chart

        Parameters
        ----------
        y : list, np.ndarray
            Input data (1-D)
        std : int, float, optional
            Number of standard deviations used to calculate if a y-value is
            out-of-control.
        ucl_limit : float, optional
            Limit the upper control limit to this value
        lcl_limit : float, optional
            Limit the lower control limit to this value
        plot_title : str, optional
            Over-ride the plot title
        """

        self.y = np.array(y) if isinstance(y, list) else y
        self.std = std
        self.ucl_limit = ucl_limit
        self.lcl_limit = lcl_limit
        self.var_name = var_name
        self.x = x
        self.plot_title = (
            "Univariate Control Chart" if plot_title is None else plot_title
        )

        # since moving range is calculated based on 2 consecutive points
        self.scalar_d = 1.128

        self.plots = []

    def __str__(self):
        msg = [
            "center_line: %0.3f" % self.center_line,
            "control_limits: %0.3f, %0.3f" % self.control_limits,
            "out_of_control: %s" % self.out_of_control,
        ]
        return "\n".join(msg)

    def __repr__(self):
        return str(self)

    @property
    def y_no_nan(self):
        return self.y[~np.isnan(self.y)]

    @property
    def center_line(self):
        """Center line"""
        return np.mean(self.y_no_nan)

    @property
    def avg_moving_range(self):
        """Avg moving range based on 2 consecutive points"""
        return np.mean(np.absolute(np.diff(self.y_no_nan)))

    @property
    def sigma(self):
        """UCL/LCL = center_line +/- sigma * std"""
        return self.avg_moving_range / self.scalar_d

    @property
    def control_limits(self):
        """Calculate the lower and upper control limits

        Returns
        ----------
        float, float
            A tuple is returned: lower control limit and upper control limit
        """
        cl = self.center_line
        sigma = self.sigma

        ucl = cl + self.std * sigma
        lcl = cl - self.std * sigma

        if self.ucl_limit is not None and ucl > self.ucl_limit:
            ucl = self.ucl_limit
        if self.lcl_limit is not None and lcl < self.lcl_limit:
            lcl = self.lcl_limit

        return lcl, ucl

    @property
    def out_of_control(self):
        """Get the indices of out-of-control observations

        Returns
        ----------
        np.ndarray
            An array of indices that are not between the lower and upper
            control limits
        """
        lcl, ucl = self.control_limits
        high = np.argwhere(self.y > ucl)
        low = np.argwhere(self.y < lcl)
        return np.unique(np.concatenate([high, low]))

    @property
    def out_of_control_high(self):
        """Get the indices of observations > ucl"""
        _, ucl = self.control_limits
        return np.argwhere(self.y > ucl)

    @property
    def out_of_control_low(self):
        """Get the indices of observations < lcl"""
        lcl, _ = self.control_limits
        return np.argwhere(self.y < lcl)

    def show(self):
        """Display the univariate control chart with matplotlib"""
        lcl, ucl = self.control_limits
        self.plots.append(
            plot.ControlChart(
                self.y,
                self.out_of_control,
                self.center_line,
                x=self.x,
                title=self.plot_title,
                lcl=lcl,
                ucl=ucl,
                ylabel=self.var_name,
            )
        )
        return self.plots[-1].figure.number

    def close(self, figure_number):
        """Close a plot by figure_number"""
        close_plot(figure_number, self.plots)


class HotellingT2:
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(self, data, alpha=0.05, plot_title=None):
        """Initialize the Hotelling T^2 class

        Parameters
        ----------
        data : np.ndarray
            A 2-D array of data to perform multivariate analysis.
            (e.g., DVHAStats.data)
        alpha : float
            The significance level used to calculate the
            upper control limit (UCL)
        plot_title : str, optional
            Over-ride the plot title
        """

        self.data = data
        self.alpha = alpha
        self.lcl = 0
        self.plot_title = (
            "Multivariate Control Chart" if plot_title is None else plot_title
        )
        self.plots = []

    def __str__(self):
        msg = [
            "Q: %s" % self.Q,
            "center_line: %0.3f" % self.center_line,
            "control_limits: %d, %0.3f" % (self.lcl, self.ucl),
            "out_of_control: %s" % self.out_of_control,
        ]
        return "\n".join(msg)

    def __repr__(self):
        return str(self)

    @property
    def observations(self):
        """Number of observations"""
        return self.data.shape[0]

    @property
    def variable_count(self):
        """Number of variables"""
        return self.data.shape[1]

    @property
    def Q(self):
        """Calculate Hotelling T^2 from a 2-D numpy array

        Returns
        -------
        np.ndarray
            A numpy array of Hotelling T^2 (1-D of length N)
        """
        Q = np.zeros(np.size(self.data, 0))
        D_bar = np.mean(self.data, axis=0)
        S = np.cov(self.data.T)
        S_inv = np.linalg.inv(S)
        observations = np.size(self.data, 0)
        for i in range(observations):
            spread = self.data[i, :] - D_bar
            Q[i] = np.matmul(np.matmul(spread, S_inv), spread)
        return Q

    @property
    def center_line(self):
        """Center line"""
        return self.get_control_limit(0.5)

    @property
    def control_limits(self):
        """Lower and Upper control limits"""
        return self.lcl, self.ucl

    @property
    def ucl(self):
        """Upper control limit"""
        return self.get_control_limit(1 - self.alpha / 2)

    @property
    def out_of_control(self):
        """Indices of out-of-control observations"""
        return np.argwhere(self.Q > self.ucl).T[0]

    def get_control_limit(self, x):
        """Calculate a Hotelling T^2 control limit using a beta distribution

        Parameters
        ----------
        x : float
            Value where the beta function is evaluated

        Returns
        -------
        float
            The control limit for a beta distribution
        """

        N = self.observations
        a = self.variable_count / 2
        b = (N - self.variable_count - 1) / 2
        return ((N - 1) ** 2 / N) * beta.ppf(x, a, b)

    def show(self):
        """Display the multivariate control chart with matplotlib"""
        self.plots.append(
            plot.ControlChart(
                self.Q,
                self.out_of_control,
                self.center_line,
                title=self.plot_title,
                ucl=self.ucl,
                lcl=self.lcl,
                ylabel="Hottelling T^2",
            )
        )
        return self.plots[-1].figure.number

    def close(self, figure_number):
        """Close a plot by figure_number"""
        close_plot(figure_number, self.plots)
