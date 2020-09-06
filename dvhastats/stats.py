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
from dvhastats.utilities import dict_to_array, csv_to_dict, moving_avg
from dvhastats import plot
from scipy import stats as scipy_stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA as sklearnPCA
from regressors import stats as regressors_stats


class DVHAStatsBaseClass:
    """Base Class for DVHAStats objects and child objects"""

    def __init__(self):
        """Initialization of DVHAStatsBaseClass for common attr/methods"""
        self.plots = []

    def close(self, figure_number):
        """Close a plot by figure_number"""
        for i, p in enumerate(self.plots):
            if p.figure.number == figure_number:
                p.close()
                self.plots.pop(i)
                return


class DVHAStats(DVHAStatsBaseClass):
    def __init__(
        self,
        data,
        var_names=None,
        x_axis=None,
        avg_len=5,
        del_const_vars=False,
    ):
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
        del_const_vars : bool
            Automatically delete any variables that have constant data. The
            names of these variables are stored in the excluded_vars attr.
            Default value is False.
        """
        DVHAStatsBaseClass.__init__(self)
        if isinstance(data, np.ndarray):
            self.data = data
            self.var_names = (
                var_names
                if var_names is not None
                else list(range(data.shape[1]))
            )
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

        self.deleted_vars = []

        self.box_cox_data = None

        self.avg_len = avg_len

        if del_const_vars:
            self.del_const_vars()

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

    def del_const_vars(self):
        """Permanently remove constant variables"""
        self.deleted_vars = self.constant_vars
        self.data = self.non_const_data

    def correlation_matrix(self, corr_type="Pearson"):
        """Get a Pearson-R and Spearman correlation matrices

        Parameters
        ----------
        corr_type : str
            Either "Pearson" or "Spearman"

        Returns
        ----------
        CorrelationMatrix
            A CorrelationMatrix class object
        """
        return CorrelationMatrix(
            self.data, self.var_names, corr_type=corr_type
        )

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

    def is_constant(self, var_name):
        """Determine if data by var_name is constant

        Parameters
        ----------
        var_name : int, str
            The var_name to check (or index of variable)

        Returns
        ----------
        bool
            True if all values of var_name are the same (i.e., no variation)
        """
        data = self.get_data_by_var_name(var_name)
        return np.all(data == data[0])

    @property
    def constant_vars(self):
        """Get a list of all constant variables"""
        return [v for v in self.var_names if self.is_constant(v)]

    @property
    def constant_var_indices(self):
        """Get a list of all constant variable indices"""
        return [i for i, v in enumerate(self.var_names) if self.is_constant(v)]

    @property
    def non_const_data(self):
        """Return self.data excluding any constant variables"""
        return np.delete(self.data, self.constant_var_indices, axis=1)

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
        const_policy="propagate",
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
        const_policy : str
            {‘propagate’, ‘raise’}
            Defines how to handle when data is constant. The following
            options are available (default is ‘propagate’):
            ‘propagate’: returns nan
            ‘raise’: throws an error

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
                self.box_cox(
                    alpha=box_cox_alpha,
                    lmbda=box_cox_lmbda,
                    const_policy=const_policy,
                )
            cc_data = self.box_cox_data
            plot_title = "Univariate Control Chart with Box-Cox Transformation"
        else:
            cc_data = self.data
            plot_title = None
        for i, key in enumerate(self.var_names):
            if const_policy == "propagate" and np.all(np.isnan(cc_data[:, i])):
                plot_title = "Cannot calculate control chart with const data!"
            data[key] = ControlChartData(
                cc_data[:, i], var_name=key, plot_title=plot_title, **kwargs
            )
            data[i] = data[key]
        return data

    def hotelling_t2(
        self,
        alpha=0.05,
        box_cox=False,
        box_cox_alpha=None,
        box_cox_lmbda=None,
        const_policy="omit",
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
        const_policy : str
            {‘raise’, 'omit'}
            Defines how to handle when data is constant. The following
            options are available (default is ‘raise’):
            ‘raise’: throws an error
            'omit': exclude constant variables from calculation

        Returns
        ----------
        HotellingT2
            HotellingT2 class object
        """

        if box_cox:
            if self.box_cox_data is None:
                p = "propagate" if const_policy == "omit" else const_policy
                self.box_cox(
                    alpha=box_cox_alpha, lmbda=box_cox_lmbda, const_policy=p
                )
            data = self.box_cox_data
            if const_policy == "omit":
                data = data[:, ~np.all(np.isnan(data), axis=0)]
            plot_title = (
                "Multivariate Control Chart with Box-Cox Transformation"
            )
        else:
            data = self.non_const_data if const_policy == "omit" else self.data
            plot_title = None

        return HotellingT2(data, alpha, plot_title=plot_title)

    def box_cox_by_index(
        self, index, alpha=None, lmbda=None, const_policy="propagate"
    ):
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
        const_policy : str
            {‘propagate’, ‘raise’}
            Defines how to handle when data is constant. The following
            options are available (default is ‘propagate’):
            ‘propagate’: returns nan
            ‘raise’: throws an error
        """
        if self.box_cox_data is None:
            self.box_cox_data = np.zeros_like(self.data)

        if isinstance(index, str):
            index = self.get_index_by_var_name(index)

        if const_policy == "omit":
            pass

        try:
            result, _ = scipy_stats.boxcox(
                self.data[:, index], alpha=alpha, lmbda=lmbda
            )
        except ValueError as e:
            if const_policy == "propagate":
                self.box_cox_data[:, index] = np.array(
                    [np.nan] * self.observations
                )
                return np.nan
            raise e

        self.box_cox_data[:, index] = result

        return self.box_cox_data[:, index]

    def box_cox(self, alpha=None, lmbda=None, const_policy="propagate"):
        """Apply box_cox_by_index for all data"""
        for i in range(self.variable_count):
            self.box_cox_by_index(
                i, alpha=alpha, lmbda=lmbda, const_policy=const_policy
            )

    def add_tend_line(self, var_name, plot_index):
        """Add trend line based on moving average"""
        trend_x, trend_y = moving_avg(
            self.get_data_by_var_name(var_name), self.avg_len
        )
        self.plots[plot_index].add_line(
            trend_y, x=trend_x, line_color="black", line_width=0.75
        )

    def pca(self, n_components=0.95, transform=True, **kwargs):
        """Return an sklearn PCA-like object, see PCA object for details"""
        return PCA(
            self.data,
            self.var_names,
            n_components=n_components,
            transform=transform,
            **kwargs
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
                title="",
            )
        )
        self.add_tend_line(var_name, -1)
        return self.plots[-1].figure.number


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


class ControlChartData(DVHAStatsBaseClass):
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
        DVHAStatsBaseClass.__init__(self)

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
        """String representation of ControlChartData object"""
        msg = [
            "center_line: %0.3f" % self.center_line,
            "control_limits: %0.3f, %0.3f" % self.control_limits,
            "out_of_control: %s" % self.out_of_control,
        ]
        return "\n".join(msg)

    def __repr__(self):
        """Return the string representation"""
        return str(self)

    @property
    def y_no_nan(self):
        """Remove indices with values of np.nan"""
        return self.y[~np.isnan(self.y)]

    @property
    def center_line(self):
        """Center line"""
        data = self.y_no_nan
        if len(data):
            return np.mean(self.y_no_nan)
        return np.nan

    @property
    def avg_moving_range(self):
        """Avg moving range based on 2 consecutive points"""
        data = self.y_no_nan
        if len(data):
            return np.mean(np.absolute(np.diff(self.y_no_nan)))
        return np.nan

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


class HotellingT2(DVHAStatsBaseClass):
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
        DVHAStatsBaseClass.__init__(self)

        self.data = data
        self.alpha = alpha
        self.lcl = 0
        self.plot_title = (
            "Multivariate Control Chart" if plot_title is None else plot_title
        )
        self.plots = []

    def __str__(self):
        """String representation of HotellingT2 object"""
        msg = [
            "Q: %s" % self.Q,
            "center_line: %0.3f" % self.center_line,
            "control_limits: %d, %0.3f" % (self.lcl, self.ucl),
            "out_of_control: %s" % self.out_of_control,
        ]
        return "\n".join(msg)

    def __repr__(self):
        """Return the string representation"""
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


class PCA(sklearnPCA, DVHAStatsBaseClass):
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(
        self, X, var_names=None, n_components=0.95, transform=True, **kwargs
    ):
        """Initialize PCA and perform fit. Inherits sklearn.decomposition.PCA

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        var_names : str, optional
            Names of the independent variables in X
        n_components : int, float, None or str
            Number of components to keep. if n_components is not set all
            components are kept:
                n_components == min(n_samples, n_features)

                If n_components == 'mle' and svd_solver == 'full', Minka’s MLE
                is used to guess the dimension. Use of n_components == 'mle'
                will interpret svd_solver == 'auto' as svd_solver == 'full'.

                If 0 < n_components < 1 and svd_solver == 'full', select the
                number of components such that the amount of variance that
                needs to be explained is greater than the percentage specified
                by n_components.

                If svd_solver == 'arpack', the number of components must be
                strictly less than the minimum of n_features and n_samples.
        transform : bool
            Fit the model and apply the dimensionality reduction
        kwargs : any
            Provide any keyword arguments for sklearn.decomposition.PCA:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        DVHAStatsBaseClass.__init__(self)
        self.X = X
        self.var_names = range(X.shape[1]) if var_names is None else var_names
        self.plots = []
        sklearnPCA.__init__(self, n_components=n_components, **kwargs)

        if transform:
            self.fit_transform(self.X)
        else:
            self.fit(self.X)

    @property
    def feature_map_data(self):
        return self.components_

    def show(self, plot_type="feature_map", absolute=True):
        """Create a heat map of PCA components

        Parameters
        ----------
        plot_type : str
            Select a plot type to display. Options include: feature_map.
        absolute : bool
            Heat map will display the absolute values in PCA components
            if True
        """
        if plot_type == "feature_map":
            data = self.feature_map_data
            if absolute:
                data = abs(data)
            self.plots.append(plot.PCAFeatureMap(data, self.var_names))
            return self.plots[-1].figure.number


class CorrelationMatrix(DVHAStatsBaseClass):
    """Pearson-R correlation matrix"""

    def __init__(
        self, X, var_names=None, corr_type="Pearson", cmap="coolwarm"
    ):
        """Initialization of CorrelationMatrix object

        Parameters
        ----------
        X : np.ndarray
            Input data (2-D) with N rows of observations and
            p columns of variables.
        corr_type : str
            Either "Pearson" or "Spearman"
        cmap : str
            matplotlib compatible color map
        """
        DVHAStatsBaseClass.__init__(self)
        self.X = X
        self.var_names = range(X.shape[1]) if var_names is None else var_names
        self.corr_type = corr_type.lower()
        self.cmap = cmap

        if self.corr_type not in {"pearson", "spearman"}:
            msg = "Invalid corr_type: must be either 'Pearson' or 'Spearman'"
            raise NotImplementedError(msg)

        func_map = {
            "pearson": pearson_correlation_matrix,
            "spearman": spearman_correlation_matrix,
        }
        if self.corr_type in list(func_map):
            self.corr, self.p = func_map[self.corr_type](self.X)

    def __title(self, is_corr=True):
        """Get plot title

        Parameters
        ----------
        is_corr : bool
            Set to True if plot data type is correlation, False if p-value
        """
        mat_type = "Pearson-R" if self.corr_type == "pearson" else "Spearman"
        value_type = ["p-value", "Correlation"][is_corr]
        return "%s %s Matrix" % (mat_type, value_type)

    def show(self, absolute=False, corr=True):
        """Create a heat map of PCA components

        Parameters
        ----------
        absolute : bool
            Heat map will display the absolute values in PCA components
            if True
        corr : bool
            Plot a p-value matrix if False, correlation matrix if True.
        """

        data = self.corr if corr else self.p
        data = abs(data) if absolute else data
        self.plots.append(
            plot.HeatMap(
                data,
                xlabels=self.var_names,
                ylabels=self.var_names,
                cmap=self.cmap,
                title=self.__title(corr),
            )
        )
        return self.plots[-1].figure.number


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
    np.ndarray, np.ndarray, np.ndarray
         A tuple of arrays: p-values, standard errors, and t-values
    """

    newX = np.append(np.ones((len(X), 1)), X, axis=1)
    mse = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

    variance = mse * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    std_errs = np.sqrt(variance)

    params = np.append(y_intercept, slope)
    t_values = np.array(params) / std_errs

    p_value = np.array(
        [
            2 * (1 - scipy_stats.t.cdf(np.abs(i), (len(newX) - 1)))
            for i in t_values
        ]
    )

    return p_value, std_errs, t_values


def pearson_correlation_matrix(X):
    """Calculate a correlation matrix of Pearson-R values

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

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


def spearman_correlation_matrix(X, nan_policy="omit"):
    """Calculate a Spearman correlation matrix

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
    nan_policy : str
        Value must be one of the following: ‘propagate’, ‘raise’, ‘omit’
        Defines how to handle when input contains nan. The following options
        are available (default is ‘omit’):
        ‘propagate’: returns nan
        ‘raise’: throws an error
        ‘omit’: performs the calculations ignoring nan values
    """
    return scipy_stats.spearmanr(X, nan_policy=nan_policy)
