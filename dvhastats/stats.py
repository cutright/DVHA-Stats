#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# stats.py
"""Statistical calculations and class objects"""
#
# Copyright (c) 2020 Dan Cutright
# Copyright (c) 2020 Arka Roy
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


import numpy as np
from scipy import stats as scipy_stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA as sklearn_PCA
from regressors import stats as regressors_stats
from prettytable import PrettyTable


class Histogram:
    """Basic histogram plot using matplotlib histogram calculation"""

    def __init__(self, y, bins, nan_policy="omit"):
        """Initialization of Histogram class object

        y: array-like
            Input array.
        bins : int, list, str, optional
            If bins is an int, it defines the number of equal-width bins in
            the given range (10, by default). If bins is a sequence, it
            defines a monotonically increasing array of bin edges, including
            the rightmost edge, allowing for non-uniform bin widths.
            If bins is a string, it defines the method used to calculate the
            optimal bin width, as defined by histogram_bin_edges.
            ‘auto’ - Maximum of the ‘sturges’ and ‘fd’ estimators.
            Provides good all around performance.
            ‘fd’  - (Freedman Diaconis Estimator) Robust (resilient to
            outliers) estimator that takes into account data variability and
            data size.
            ‘doane’ - An improved version of Sturges’ estimator that works
            better with non-normal datasets.
            ‘scott’ - Less robust estimator that that takes into account data
            variability and data size.
            ‘stone’ - Estimator based on leave-one-out cross-validation
            estimate of the integrated squared error. Can be regarded as a
            generalization of Scott’s rule.
            ‘rice’ - Estimator does not take variability into account, only
            data size. Commonly overestimates number of bins required.
            ‘sturges’ - R’s default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins for
            large non-gaussian datasets.
            ‘sqrt’ - Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.
        nan_policy : str
            Value must be one of the following: ‘propagate’, ‘raise’, ‘omit’
            Defines how to handle when input contains nan. The following
            options are available (default is ‘omit’):
            ‘propagate’: returns nan
            ‘raise’: throws an error
            ‘omit’: performs the calculations ignoring nan values
        """

        self.y = process_nan_policy(y, nan_policy)
        self.bins = bins
        self.nan_policy = nan_policy

    @property
    def mean(self):
        """The mean value of the input array

        Returns
        ----------
        np.ndarray
            Mean value of y with np.mean()
        """
        return np.mean(self.y)

    @property
    def median(self):
        """The median value of the input array

        Returns
        ----------
        np.ndarray
            Median value of y with np.median()
        """
        return np.median(self.y)

    @property
    def std(self):
        """The standard deviation of the input array

        Returns
        ----------
        np.ndarray
            Standard deviation of y with np.std()
        """
        return np.std(self.y)

    @property
    def normality(self):
        """The normality and normality p-value of the input array

        Returns
        ----------
        statistic : float
            Normality calculated with scipy.stats.normaltest
        p-value : float
            A 2-sided chi squared probability for the hypothesis test.
        """
        return scipy_stats.normaltest(self.y, nan_policy=self.nan_policy)

    @property
    def hist_data(self):
        """Get the histogram data

        Returns
        ----------
        hist : np.ndarray
            The values of the histogram
        center : np.ndarray
            The centers of the bins
        """
        hist, bin_edges = np.histogram(self.y, bins=self.bins)
        center = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        return hist, center

    @property
    def chart_data(self):
        """JSON compatible dict for chart generation

        Returns
        ----------
        dict
            Data used for Histogram visuals. Keys include 'x', 'y', 'mean',
            'median', 'std', 'normality', 'normality_p'
        """
        hist, center = self.hist_data
        norm, norm_p = self.normality
        return {
            "x": center.tolist(),
            "y": hist.tolist(),
            "mean": float(self.mean),
            "median": float(self.median),
            "std": float(self.std),
            "normality": float(norm),
            "normality_p": float(norm_p),
        }


class MultiVariableRegression:
    """Multi-variable regression using scikit-learn"""

    def __init__(self, X, y, saved_reg=None, var_names=None):
        """Initialization of a MultiVariableRegression

        Parameters
        ----------
        X : array-like
            Independent data
        y : array-like
            Dependent data
        saved_reg : MultiVariableRegression, optional
            Optionally provide a previously calculated regression
        var_names : list, optional
            Optionally provide names of the variables
        """

        self.X = np.array(X)
        self.y = np.array(y)

        self.var_names = (
            list(range(self.X.shape[1])) if var_names is None else var_names
        )

        self._do_fit(saved_reg=saved_reg)

    def __str__(self):
        """String representation of MultiVariableRegression object"""

        table = PrettyTable()

        table.field_names = ["", "Coef", "Std. Err.", "t-value", "p-value"]
        for c in table.field_names[1:]:
            table.align[c] = 'r'
        data = {"": ["y-int"] + self.var_names,
                "Coef": ["%0.3E" % v for v in self.coef],
                "Std. Err.": ["%0.3E" % v for v in self.std_err],
                "t-value": ["%0.3f" % v for v in self.t],
                "p-value": ["%0.3f" % v for v in self.p]}

        for r in range(0, len(data[""])):
            table.add_row([data[c][r] for c in table.field_names])

        msg = [
            "Multi-Variable Regression results/model\n",
            "R²: %0.3f" % self.r_sq,
            "MSE: %0.3f" % self.mse,
            "f-stat: %0.3f" % self.f_stat,
            "f-stat p-value: %0.3f" % self.f_p_value,
            str(table)
        ]
        return "\n".join(msg)

    def __repr__(self):
        """Return the string representation"""
        return str(self)

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

    @property
    def y_intercept(self):
        """The y-intercept of the linear regression

        Returns
        ----------
        float
            Independent term in the linear model.
        """
        return self.reg.intercept_ if self.reg is not None else None

    @property
    def slope(self):
        """The slope of the linear regression

        Returns
        ----------
        np.ndarray
            Estimated coefficients for the linear regression problem. If
            multiple targets are passed during the fit (y 2D), this is a 2D
            array of shape (n_targets, n_features), while if only one target
            is passed, this is a 1D array of length n_features.
        """
        return self.reg.coef_ if self.reg is not None else None

    @property
    def r_sq(self):
        """R^2 (coefficient of determination) regression score function.

        Returns
        ----------
        float
            The R^2 score
        """
        return r2_score(self.y, self.predictions)

    @property
    def mse(self):
        """Mean squared error of the linear regression

        Returns
        ----------
        float, nd.array
            A non-negative floating point value (the best value is 0.0), or
            an array of floating point values, one for each individual target.
        """
        return mean_squared_error(self.y, self.predictions)

    @property
    def residuals(self):
        """Residuals of the prediction and sample data

        Returns
        ----------
        np.ndarray
            y - predictions
        """
        return np.subtract(self.y, self.predictions)

    @property
    def f_stat(self):
        """The F-statistic of the regression

        Returns
        ----------
        float
            F-statistic of beta coefficients using regressors.stats
        """
        return regressors_stats.f_stat(self.ols, self.X, self.y)

    @property
    def df_error(self):
        """Error degrees of freedom

        Returns
        ----------
        int
            Degrees of freedom for the error
        """
        return len(self.X[:, 0]) - len(self.X[0, :]) - 1

    @property
    def df_model(self):
        """Model degrees of freedom

        Returns
        ----------
        int
            Degrees of freedom for the model
        """
        return len(self.X[0, :])

    @property
    def f_p_value(self):
        """p-value of the f-statistic

        Returns
        ----------
        float
            p-value of the F-statistic of beta coefficients using scipy
        """
        return scipy_stats.f.cdf(self.f_stat, self.df_model, self.df_error)

    @property
    def coef(self):
        """Coefficients for the regression

        Returns
        ----------
        np.ndarray
            An array of regression coefficients (i.e., y_intercept,
            1st var slope, 2nd var slope, etc.)
        """
        return np.concatenate(([self.y_intercept], self.slope))

    @property
    def chart_data(self):
        """JSON compatible dict for chart generation

        Returns
        ----------
        dict
            Data used for residual visuals. Keys include 'x', 'y', 'pred',
            'resid', 'coef', 'r_sq', 'mse', 'std_err', 't_value', 'p_value'
        """
        return {
            "y": self.y.tolist(),
            "pred": self.predictions.tolist(),
            "resid": self.residuals.tolist(),
            "coef": self.coef.tolist(),
            "r_sq": float(self.r_sq),
            "mse": float(self.mse),
            "std_err": self.std_err.tolist(),
            "t_value": self.t.tolist(),
            "p_value": self.p.tolist(),
        }

    @property
    def prob_plot(self):
        """
        Calculate quantiles for a probability plot

        Returns
        ----------
        dict
            Data for generating a probablity plot. Keys include:
            'x', 'y', 'y_intercept', 'slope', 'x_trend', 'y_trend'
        """
        norm_prob_plot = scipy_stats.probplot(
            self.residuals, dist="norm", fit=False, plot=None, rvalue=False
        )

        reg_prob = linear_model.LinearRegression()
        reg_prob.fit([[val] for val in norm_prob_plot[0]], norm_prob_plot[1])

        x_trend = [
            float(np.min(norm_prob_plot[0])),
            float(np.max(norm_prob_plot[0])),
        ]
        y_trend = np.add(
            np.multiply(x_trend, reg_prob.coef_),
            reg_prob.intercept_,
        )
        return {
            "x": norm_prob_plot[0].tolist(),
            "y": norm_prob_plot[1].tolist(),
            "y_intercept": float(reg_prob.intercept_),
            "slope": float(reg_prob.coef_),
            "x_trend": x_trend,
            "y_trend": y_trend.tolist(),
        }


class ControlChart:
    """Univariate Control Chart"""

    def __init__(
        self,
        y,
        std=3,
        ucl_limit=None,
        lcl_limit=None,
        x=None,
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
        """

        self.y = np.array(y) if isinstance(y, list) else y
        self.x = x if x else np.linspace(1, len(self.y), len(self.y))
        self.std = std
        self.ucl_limit = ucl_limit
        self.lcl_limit = lcl_limit

        # since moving range is calculated based on 2 consecutive points
        self.scalar_d = 1.128

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
    def center_line(self):
        """Center line of charting data (i.e., mean value)

        Returns
        ----------
        np.ndarray, np.nan
            Mean value of y with np.mean() or np.nan if y is empty
        """
        data = remove_nan(self.y)
        if len(data):
            return np.mean(data)
        return np.nan

    @property
    def avg_moving_range(self):
        """Avg moving range based on 2 consecutive points

        Returns
        ----------
        np.ndarray, np.nan
            Average moving range. Returns NaN if arr is empty.
        """
        return avg_moving_range(self.y, nan_policy="omit")

    @property
    def sigma(self):
        """UCL/LCL = center_line +/- sigma * std

        Returns
        ----------
        np.ndarray, np.nan
            sigma or np.nan if arr is empty
        """
        return self.avg_moving_range / self.scalar_d

    @property
    def control_limits(self):
        """Calculate the lower and upper control limits

        Returns
        ----------
        lcl : float
            Lower Control Limit (LCL)
        ucl : float
            Upper Control Limit (UCL)
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
        """Get the indices of observations > ucl

        Returns
        ----------
        np.ndarray
            An array of indices that are greater than the upper control limit
        """
        _, ucl = self.control_limits
        return np.argwhere(self.y > ucl)

    @property
    def out_of_control_low(self):
        """Get the indices of observations < lcl

        Returns
        ----------
        np.ndarray
            An array of indices that are less than the lower control limit
        """
        lcl, _ = self.control_limits
        return np.argwhere(self.y < lcl)

    @property
    def chart_data(self):
        """JSON compatible dict for chart generation

        Returns
        ----------
        dict
            Data used for Histogram visuals. Keys include 'x', 'y',
            'out_of_control', 'center_line', 'lcl', 'ucl'
        """
        lcl, ucl = self.control_limits
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "out_of_control": self.out_of_control.tolist(),
            "center_line": float(self.center_line),
            "lcl": float(lcl),
            "ucl": float(ucl),
        }


class HotellingT2:
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(self, data, alpha=0.05, const_policy="raise"):
        """Initialize the Hotelling T^2 class

        Parameters
        ----------
        data : np.ndarray
            A 2-D array of data to perform multivariate analysis.
            (e.g., DVHAStats.data)
        alpha : float
            The significance level used to calculate the
            upper control limit (UCL)
        const_policy : str
            {‘raise’, 'omit'}
            Defines how to handle when data is constant. The following
            options are available (default is ‘raise’):
            ‘raise’: throws an error
            'omit': exclude constant variables from calculation
        """

        self.data = (
            data if const_policy == "raise" else remove_const_column(data)
        )
        self.alpha = alpha
        self.lcl = 0

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

    @property
    def Q(self):
        """Calculate Hotelling T^2 statistic (Q) from a 2-D numpy array

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
        """Center line for the control chart

        Returns
        ----------
        float
            Median value of beta distribution.
        """
        return self.get_control_limit(0.5)

    @property
    def control_limits(self):
        """Lower and Upper control limits

        Returns
        ----------
        lcl : float
            Lower Control Limit (LCL). This is fixed to 0 for Hotelling T2
        ucl : float
            Upper Control Limit (UCL)
        """
        return self.lcl, self.ucl

    @property
    def ucl(self):
        """Upper control limit

        Returns
        ----------
        ucl : float
            Upper Control Limit (UCL)"""
        return self.get_control_limit(1 - self.alpha / 2)

    @property
    def out_of_control(self):
        """Indices of out-of-control observations

        Returns
        ----------
        np.ndarray
            An array of indices that are greater than the upper
            control limit. (NOTE: Q is never negative)
        """
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
        return ((N - 1) ** 2 / N) * scipy_stats.beta.ppf(x, a, b)

    @property
    def chart_data(self):
        """JSON compatible dict for chart generation

        Returns
        ----------
        dict
            Data used for Histogram visuals. Keys include 'x', 'y',
            'out_of_control', 'center_line', 'lcl', 'ucl'
        """
        return {
            "x": list(range(1, self.observations + 1)),
            "y": self.Q.tolist(),
            "out_of_control": self.out_of_control.tolist(),
            "center_line": float(self.center_line),
            "lcl": float(self.lcl),
            "ucl": float(self.ucl),
        }


class PCA(sklearn_PCA):
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(
        self, X, var_names=None, n_components=0.95, transform=True, **kwargs
    ):
        """Initialize PCA and perform fit. Inherits sklearn.decomposition.PCA

        Parameters
        ----------
        X : np.ndarray
            Training data (2-D), where n_samples is the number of samples and
            n_features is the number of features. shape (n_samples, n_features)
        var_names : list, optional
            Optionally provide names of the features
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

        self.X = X
        self.var_names = (
            list(range(self.X.shape[1])) if var_names is None else var_names
        )
        sklearn_PCA.__init__(self, n_components=n_components, **kwargs)

        if transform:
            self.fit_transform(self.X)
        else:
            self.fit(self.X)

    @property
    def feature_map_data(self):
        """Used for feature analysis heat map

        Returns
        ----------
        np.ndarray Shape (n_components, n_features)
            Principal axes in feature space, representing the directions of
            maximum variance in the data. The components are sorted by
            explained_variance.
        """
        return self.components_

    @property
    def component_labels(self):
        """Get component names

        Returns
        ----------
        list
            Labels for plotting. (1st Comp, 2nd Comp, 3rd Comp, etc.)
        """
        return [
            "%s Comp" % (get_ordinal(n + 1))
            for n in range(self.components_.shape[0])
        ]


class CorrelationMatrix:
    """Pearson-R correlation matrix"""

    def __init__(self, X, corr_type="Pearson"):
        """Initialization of CorrelationMatrix object

        Parameters
        ----------
        X : np.ndarray
            Input data (2-D) with N rows of observations and
            p columns of variables.
        corr_type : str
            Either "Pearson" or "Spearman"
        """
        self.X = X
        self.corr_type = corr_type.lower()

        if self.corr_type not in {"pearson", "spearman"}:
            msg = "Invalid corr_type: must be either 'Pearson' or 'Spearman'"
            raise NotImplementedError(msg)

        func_map = {
            "pearson": pearson_correlation_matrix,
            "spearman": spearman_correlation_matrix,
        }
        if self.corr_type in list(func_map):
            self.corr, self.p = func_map[self.corr_type](self.X)

    @property
    def normality(self):
        """The normality and normality p-value of the input array

        Returns
        ----------
        statistic : np.ndarray
            Normality calculated with scipy.stats.normaltest
        p-value : np.ndarray
            A 2-sided chi squared probability for the hypothesis test.
        """
        return scipy_stats.normaltest(self.X, nan_policy="omit")

    @property
    def chart_data(self):
        """JSON compatible dict for chart generation

        Returns
        ----------
        dict
            Data used for Histogram visuals. Keys include 'corr', 'p',
            'norm', 'norm_p'
        """
        norm, norm_p = self.normality
        return {
            "corr": self.corr.tolist(),
            "p": self.p.tolist(),
            "norm": norm.tolist(),
            "norm_p": norm_p.tolist(),
        }


#########################################################
# Pure stats functions
#########################################################


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
    y_intercept : float, np.ndarray
        The y-intercept of the linear regression
    slope : float, np.ndarray
        The slope of the linear regression

    Returns
    ----------
    p_value : np.ndarray
        p-value of the linear regression coefficients
    std_errs : np.ndarray
        standard errors of the linear regression coefficients
    t_value : np.ndarray
        t-values of the linear regression coefficients
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

    Returns
    ----------
    r : np.ndarray
        Array (2-D) of Pearson-R correlations between the row indexed and
        column indexed variables
    p : np.ndarray
        Array (2-D) of p-values associated with r
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

    Returns
    ----------
    correlation : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2
        variables are given as parameters. Correlation matrix is square with
        length equal to total number of variables (columns or rows) in a and
        b combined.
    p-value : float
        The two-sided p-value for a hypothesis test whose null hypothesis is
        that two sets of data are uncorrelated, has same dimension as rho.

    """
    return scipy_stats.spearmanr(X, nan_policy=nan_policy)


def moving_avg(y, avg_len, x=None, weight=None):
    """Calculate the moving (rolling) average of a set of data

    Parameters
    ----------
    y : array-like
        data (1-D) to be averaged
    avg_len : int
        Data is averaged over this many points (current value and avg_len - 1
        prior points)
    x : np.ndarray, list, optional
        Optionally specify the x-axis values. Otherwise index+1 is used.
    weight : np.ndarray, list, optional
        A weighted moving average is calculated based on the provided weights.
        weight must be of same length as y. Weights of one are assumed by
        default.

    Returns
    ----------
    x : np.ndarray
        Resulting x-values for the moving average
    moving_avg : np.ndarray
        moving average values
    """
    x = np.linspace(1, len(y), len(y)) if x is None else x
    weight = np.ones_like(y) if weight is None else weight

    cumsum, moving_aves, x_final = [0], [], []

    for i, yi in enumerate(y, 1):
        cumsum.append(cumsum[i - 1] + yi / weight[i - 1])
        if i >= avg_len:
            moving_ave = (cumsum[i] - cumsum[i - avg_len]) / avg_len
            moving_aves.append(moving_ave)
    x_final = [x[i] for i in range(avg_len - 1, len(x))]

    return np.array(x_final), np.array(moving_aves)


def box_cox(arr, alpha=None, lmbda=None, const_policy="propagate"):
    """

    Parameters
    ----------
    arr : np.ndarray
        Input array. Must be positive 1-dimensional.
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

    Returns
    ----------
    box_cox : np.ndarray
        Box-Cox power transformed array
    """

    try:
        box_cox_data, _ = scipy_stats.boxcox(arr, alpha=alpha, lmbda=lmbda)
    except ValueError as e:
        if const_policy == "propagate":
            return np.array([np.nan] * arr.shape[0])
        raise e

    return box_cox_data


def avg_moving_range(arr, nan_policy="omit"):
    """Calculate the average moving range (over 2-consecutive point1)

    Parameters
    ----------
    arr : array-like (1-D)
        Input array. Must be positive 1-dimensional.
    nan_policy : str, optional
        Value must be one of the following: {‘propagate’, ‘raise’, ‘omit’}
        Defines how to handle when input contains nan. The following options
        are available (default is ‘omit’):
        ‘propagate’: returns nan
        ‘raise’: throws an error
        ‘omit’: performs the calculations ignoring nan values

    Returns
    ----------
    np.ndarray, np.nan
        Average moving range. Returns NaN if arr is empty
    """

    arr = process_nan_policy(arr, nan_policy)
    if len(arr) == 0:
        return np.nan
    return np.mean(np.absolute(np.diff(arr)))


###################################################
# Stats related utilities
###################################################


def remove_nan(arr):
    """Remove indices from 1-D array with values of np.nan

    Parameters
    ----------
    arr : np.ndarray (1-D)
        Input array. Must be positive 1-dimensional.

    Returns
    ----------
    np.ndarray
        arr with NaN values deleted

    """
    return arr[~np.isnan(arr)]


def remove_const_column(arr):
    """Remove all columns with zero variance

    Parameters
    ----------
    arr : np.ndarray
        Input array (2-D)

    Returns
    ----------
    np.ndarray
        Input array with columns of a constant value removed
    """
    return arr[:, ~np.all(np.isnan(arr), axis=0)]


def is_nan_arr(arr):
    """Check if array has only NaN elements

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    ----------
    bool
        True if all elements are np.nan
    """
    return np.all(np.isnan(arr))


def is_arr_constant(arr):
    """Determine if data by var_name is constant

    Parameters
    ----------
    arr : array-like
        Input array or object that can be converted to an array

    Returns
    ----------
    bool
        True if all values the same (i.e., no variation)
    """
    return np.all(arr == arr[0])


def process_nan_policy(arr, nan_policy):
    """Calculate the average moving range (over 2-consecutive point1)

    Parameters
    ----------
    arr : array-like (1-D)
        Input array. Must be positive 1-dimensional.
    nan_policy : str
        Value must be one of the following: {‘propagate’, ‘raise’, ‘omit’}
        Defines how to handle when input contains nan. The following options
        are available (default is ‘omit’):
        ‘propagate’: returns nan
        ‘raise’: throws an error
        ‘omit’: performs the calculations ignoring nan values

    Returns
    ----------
    np.ndarray, np.nan
        Input array evaluated per nan_policy
    """

    arr_no_nan = remove_nan(arr)
    if len(arr_no_nan) != len(arr):
        if nan_policy == "raise":
            msg = "NaN values are not supported for avg_moving_range"
            raise NotImplementedError(msg)
        if nan_policy == "propagate":
            return np.nan
        if nan_policy == "omit":
            return arr_no_nan
    return arr


def get_ordinal(n):
    """Convert number to its ordinal (e.g., 1 to 1st)

    Parameters
    ----------
    n : int
        Number to be converted to ordinal

    Returns
    ----------
    str
        the ordinal of n
    """
    return "%d%s" % (
        n,
        "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
    )
