#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ui.py
"""DVHA-Stats classes for user interaction"""
#
# Copyright (c) 2020 Dan Cutright
# Copyright (c) 2020 Arka Roy
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


from os.path import dirname, join
import numpy as np
from dvhastats.utilities import import_data
from dvhastats import plot
from dvhastats import stats

SCRIPT_DIR = dirname(__file__)
PARENT_DIR = dirname(SCRIPT_DIR)
TEST_DATA_PATH = join(PARENT_DIR, 'tests', 'testdata', 'multivariate_data.csv')


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
    """The main UI class object for DVHAStats"""

    def __init__(
        self,
        data=None,
        var_names=None,
        x_axis=None,
        avg_len=5,
        del_const_vars=False,
    ):
        """Class used to calculated various statistics

        Parameters
        ----------
        data : numpy.array, dict, str, None
            Input data (2-D) with N rows of observations and
            p columns of variables.  The CSV file must have a header row
            for column names. Test data is loaded if None
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

        data = TEST_DATA_PATH if data is None else data
        self.data, self.var_names = import_data(data, var_names)

        self.x_axis = x_axis

        self.deleted_vars = []

        self.box_cox_data = None

        self.avg_len = avg_len

        if del_const_vars:
            self.del_const_vars()

    def get_data_by_var_name(self, var_name):
        """Get the single variable array based on var_name

        Parameters
        ----------
        var_name : int, str
            The name (str) or index (int) of the variable of interest

        Returns
        ----------
        np.ndarray
            The column of data for the given var_name
        """
        index = self.get_index_by_var_name(var_name)
        return self.data[:, index]

    def get_index_by_var_name(self, var_name):
        """Get the variable index by var_name

        Parameters
        ----------
        var_name : int, str
            The name (str) or index (int) of the variable of interest

        Returns
        ----------
        int
            The column index for the given var_name
        """
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
        """Number of observations in data

        Returns
        ----------
        int
            Number of rows in data
        """
        return self.data.shape[0]

    @property
    def variable_count(self):
        """Number of variables in data

        Returns
        ----------
        int
            Number of columns in data"""
        return self.data.shape[1]

    def del_const_vars(self):
        """Permanently remove variables with no variation"""
        self.deleted_vars = self.constant_vars
        self.data = self.non_const_data

    def correlation_matrix(self, corr_type="Pearson"):
        """Get a Pearson-R or Spearman correlation matrices

        Parameters
        ----------
        corr_type : str
            Either "Pearson" or "Spearman"

        Returns
        ----------
        CorrelationMatrixUI
            A CorrelationMatrixUI class object
        """
        return CorrelationMatrixUI(
            self.data, self.var_names, corr_type=corr_type
        )

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
        return stats.is_arr_constant(data)

    @property
    def constant_vars(self):
        """Get a list of all constant variables

        Returns
        ----------
        list
            Names of variables with no variation
        """
        return [v for v in self.var_names if self.is_constant(v)]

    @property
    def constant_var_indices(self):
        """Get a list of all constant variable indices

        Returns
        ----------
        list
            Indices of variables with no variation
        """
        return [i for i, v in enumerate(self.var_names) if self.is_constant(v)]

    @property
    def non_const_data(self):
        """Return self.data excluding any constant variables

        Returns
        ----------
        np.ndarray
            Data with constant variables removed. This does not alter the data
            property.
        """
        return np.delete(self.data, self.constant_var_indices, axis=1)

    def histogram(self, var_name, bins='auto', nan_policy="omit"):
        """Get a Histogram class object

        var_name : str, int
            The name (str) or index (int) of teh variable to plot
        bins : int, list, str, optional
            See https://numpy.org/doc/stable/reference/generated/numpy.histogram.html for details
        nan_policy : str
            Value must be one of the following: ‘propagate’, ‘raise’, ‘omit’
            Defines how to handle when input contains nan. The following
            options are available (default is ‘omit’):
            ‘propagate’: returns nan
            ‘raise’: throws an error
            ‘omit’: performs the calculations ignoring nan values
        """
        data = self.get_data_by_var_name(var_name)
        return stats.Histogram(data, bins, nan_policy)

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
        LinearRegUI
            A LinearRegUI class object.
        """
        return LinearRegUI(self.data, y, saved_reg, self.var_names)

    def univariate_control_chart(self,
        var_name,
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
        var_name : str, int
            The name (str) or index (int) of teh variable to plot
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
            {‘propagate’, ‘raise’, 'omit'}
            Defines how to handle when data is constant. The following
            options are available (default is ‘propagate’):
            ‘propagate’: returns nan
            ‘raise’: throws an error
            'omit': remove NaN data

        Returns
        ----------
        stats.ControlChart
            stats.ControlChart class object
        """
        kwargs = {"std": std, "ucl_limit": ucl_limit, "lcl_limit": lcl_limit}

        index = self.get_index_by_var_name(var_name)
        var_name = self.var_names[index]

        if box_cox:
            if self.box_cox_data is None:
                cc_data = self.box_cox_by_index(
                    index,
                    alpha=box_cox_alpha,
                    lmbda=box_cox_lmbda,
                    const_policy=const_policy,
                )
            else:
                cc_data = self.box_cox_data[:, index]
            plot_title = "Univariate Control Chart with Box-Cox Transformation"
        else:
            cc_data = self.data[:, index]
            plot_title = None

        if const_policy == "propagate" and stats.is_nan_arr(cc_data):
            plot_title = "Cannot calculate control chart with const data!"
        data = ControlChartUI(
            cc_data, var_name=var_name, plot_title=plot_title, **kwargs
        )

        return data

    def univariate_control_charts(
        self,
        **kwargs
    ):
        """
        Calculate Control charts for all variables

        Parameters
        ----------
        kwargs : any
            See univariate_control_chart for keyword parameters

        Returns
        ----------
        dict
            ControlChart class objects stored in a dictionary with
            var_names and indices as keys (can use var_name or index)
        """
        data = {}
        for i, key in enumerate(self.var_names):
            data[key] = self.univariate_control_chart(key, **kwargs)
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
        HotellingT2UI
            HotellingT2UI class object
        """

        if box_cox:
            if self.box_cox_data is None:
                self.box_cox(
                    alpha=box_cox_alpha,
                    lmbda=box_cox_lmbda,
                    const_policy=const_policy,
                )
            data = self.box_cox_data
            if const_policy == "omit":
                data = stats.remove_const_column(data)
            plot_title = (
                "Multivariate Control Chart with Box-Cox Transformation"
            )
        else:
            data = self.non_const_data if const_policy == "omit" else self.data
            plot_title = None

        return HotellingT2UI(data, alpha, plot_title=plot_title)

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
            {‘propagate’, ‘raise’, 'omit'}
            Defines how to handle when data is constant. The following
            options are available (default is ‘propagate’):
            ‘propagate’: returns nan
            ‘raise’: throws an error
            'omit': remove

        Returns
        ----------
        np.ndarray
            Results from stats.box_cox
        """
        if self.box_cox_data is None:
            self.box_cox_data = np.zeros_like(self.data)

        if isinstance(index, str):
            index = self.get_index_by_var_name(index)

        self.box_cox_data[:, index] = stats.box_cox(
            self.data[:, index],
            alpha=alpha,
            lmbda=lmbda,
            const_policy=const_policy,
        )

        return self.box_cox_data[:, index]

    def box_cox(self, alpha=None, lmbda=None, const_policy="propagate"):
        """Apply box_cox_by_index for all data"""
        for i in range(self.variable_count):
            self.box_cox_by_index(
                i, alpha=alpha, lmbda=lmbda, const_policy=const_policy
            )

    def pca(self, n_components=0.95, transform=True, **kwargs):
        """Return an sklearn PCA-like object, see PCA object for details

        Parameters
        ----------
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

        Returns
        ----------
        stats.PCA
            A principal component analysis object inherited from
            sklearn.decomposition.PCA
        """
        return PCAUI(
            self.data,
            var_names=self.var_names,
            n_components=n_components,
            transform=transform,
            **kwargs
        )

    def __add_tend_line(self, var_name, plot_index):
        """Add trend line based on moving average"""
        trend_x, trend_y = stats.moving_avg(
            self.get_data_by_var_name(var_name), self.avg_len
        )
        self.plots[plot_index].add_line(
            trend_y, x=trend_x, line_color="black", line_width=0.75
        )

    def show(self, var_name, plot_type="trend", **kwargs):
        """Display a plot of var_name with matplotlib

        Parameters
        ----------
        var_name : str, int
            The name (str) or index (int) of teh variable to plot
        plot_type : str
            Either "trend" or "hist"
        kwargs : any
            If plot_type is "hist", pass any of the matplotlib hist key word
            arguments

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """
        index = self.get_index_by_var_name(var_name)
        var_name = self.var_names[index]
        if plot_type == "trend":
            self.plots.append(
                plot.Plot(
                    self.data[:, index],
                    x=self.x_axis,
                    xlabel="Observation",
                    ylabel=var_name,
                    title="",
                )
            )
            self.__add_tend_line(var_name, -1)
        elif plot_type == "hist":
            self.plots.append(
                plot.Histogram(self.data[:, index], xlabel=var_name, **kwargs)
            )
        return self.plots[-1].figure.number


class ControlChartUI(DVHAStatsBaseClass, stats.ControlChart):
    """Univariate Control Chart"""

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
        stats.ControlChart.__init__(
            self, y, std=std, ucl_limit=ucl_limit, lcl_limit=lcl_limit, x=x
        )

        self.plot_title = (
            "Univariate Control Chart" if plot_title is None else plot_title
        )

        self.var_name = var_name
        self.plots = []

    def show(self):
        """Display the univariate control chart with matplotlib

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """
        self.plots.append(
            plot.ControlChart(
                title=self.plot_title, ylabel=self.var_name, **self.chart_data
            )
        )
        return self.plots[-1].figure.number


class HotellingT2UI(DVHAStatsBaseClass, stats.HotellingT2):
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
        stats.HotellingT2.__init__(self, data, alpha=alpha)

        self.plot_title = (
            "Multivariate Control Chart" if plot_title is None else plot_title
        )
        self.plots = []

    def show(self):
        """Display the multivariate control chart with matplotlib

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """
        self.plots.append(
            plot.ControlChart(
                title=self.plot_title,
                ylabel="Hottelling T^2",
                **self.chart_data
            )
        )
        return self.plots[-1].figure.number


class PCAUI(DVHAStatsBaseClass, stats.PCA):
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
        # print(kwargs)
        DVHAStatsBaseClass.__init__(self)
        stats.PCA.__init__(
            self, X, n_components=n_components, transform=transform, **kwargs
        )
        self.var_names = range(X.shape[1]) if var_names is None else var_names
        self.plots = []

    def show(self, plot_type="feature_map", absolute=True):
        """Create a heat map of PCA components

        Parameters
        ----------
        plot_type : str
            Select a plot type to display. Options include: feature_map.
        absolute : bool
            Heat map will display the absolute values in PCA components
            if True

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """
        if plot_type == "feature_map":
            data = self.feature_map_data
            if absolute:
                data = abs(data)
            self.plots.append(plot.PCAFeatureMap(data, self.var_names))
            return self.plots[-1].figure.number


class CorrelationMatrixUI(DVHAStatsBaseClass, stats.CorrelationMatrix):
    """Pearson-R correlation matrix UI object"""

    def __init__(
        self, X, var_names=None, corr_type="Pearson", cmap="coolwarm"
    ):
        """Initialization of CorrelationMatrix object

        Parameters
        ----------
        X : np.ndarray
            Input data (2-D) with N rows of observations and
            p columns of variables.
        var_names : list, optional
            Optionally set the variable names with a list of str
        corr_type : str
            Either "Pearson" or "Spearman"
        cmap : str
            matplotlib compatible color map
        """
        DVHAStatsBaseClass.__init__(self)
        stats.CorrelationMatrix.__init__(self, X=X, corr_type=corr_type)
        self.var_names = range(X.shape[1]) if var_names is None else var_names
        self.cmap = cmap

    def show(self, absolute=False, corr=True):
        """Create a heat map of PCA components

        Parameters
        ----------
        absolute : bool
            Heat map will display the absolute values in PCA components
            if True
        corr : bool
            Plot a p-value matrix if False, correlation matrix if True.

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """

        data = self.corr if corr else self.p
        data = abs(data) if absolute else data

        mat_type = "Pearson-R" if self.corr_type == "pearson" else "Spearman"
        value_type = ["p-value", "Correlation"][corr]
        title = "%s %s Matrix" % (mat_type, value_type)

        self.plots.append(
            plot.HeatMap(
                data,
                xlabels=self.var_names,
                ylabels=self.var_names,
                cmap=self.cmap,
                title=title,
            )
        )
        return self.plots[-1].figure.number


class LinearRegUI(DVHAStatsBaseClass, stats.MultiVariableRegression):
    """A MultiVariableRegression class UI object

    Parameters
    ----------
    y : np.ndarray, list
        Dependent data based on DVHAStats.data
    saved_reg : MultiVariableRegression, optional
        If supplied, predicted values (y-hat) will be calculated with
        DVHAStats.data and the regression from saved_reg. This is useful
        if testing a regression model on new data.
    var_names : list, optional
        Optionally provide names of the variables
    """

    def __init__(self, X, y, saved_reg=None, var_names=None):
        DVHAStatsBaseClass.__init__(self)
        stats.MultiVariableRegression.__init__(
            self, X=X, y=y, saved_reg=saved_reg, var_names=var_names
        )

    def show(self, plot_type="residual"):
        """Create a Residual or Probability Plot

        Parameters
        ----------
        plot_type : str
            Either "residual" or "prob"

        Returns
        ----------
        int
            The number of the newly created matplotlib figure
        """

        if plot_type not in {"residual", "prob"}:
            return

        title = (
            "Multi-Variable Linear Regression"
            if self.X.shape[1] > 1
            else "Linear Regression"
        )

        if plot_type == "residual":
            data = self.chart_data
            self.plots.append(
                plot.Plot(
                    data["resid"],
                    x=data["y"],
                    title=title,
                    xlabel="Fitted Values",
                    ylabel="Residual",
                    line=False,
                )
            )
            x_zero = [np.min(data["y"]), np.max(data["y"])]
            y_zero = [0, 0]
            self.plots[-1].add_line(
                y_zero, x_zero, line_color="black", line_style="--"
            )
        elif plot_type == "prob":
            data = self.prob_plot
            self.plots.append(
                plot.Plot(
                    data["y"],
                    x=data["x"],
                    title="Probability Plot",
                    xlabel="Quantiles",
                    ylabel="Ordered Values",
                    line=False,
                )
            )
            self.plots[-1].add_line(
                data["y_trend"],
                data["x_trend"],
                line_color="black",
                line_style="--",
            )

        return self.plots[-1].figure.number
