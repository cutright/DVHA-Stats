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
from sklearn.decomposition import PCA as sklearnPCA
from regressors import stats as regressors_stats


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
        """p-value of the f-statistic"""
        return scipy_stats.f.cdf(self.f_stat, self.df_model, self.df_error)


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
    array-like
        x-values, moving average values
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


def normality(data, nan_policy="omit"):
    """Calculate the normality and associated p-values for each variable

    Parameters
    ----------
    data : np.ndarray
        A 2-D array of data
    nan_policy : str
        Value must be one of the following: {‘propagate’, ‘raise’, ‘omit’}
        Defines how to handle when input contains nan. The following options
        are available (default is ‘omit’):
        ‘propagate’: returns nan
        ‘raise’: throws an error
        ‘omit’: performs the calculations ignoring nan values

    Returns
    ----------
    np.ndarray, np.ndarray
        A tuple of 1-D arrays are returned: Normality and its
        p-values.
    """
    var_count = data.shape[1]
    norm, p = np.zeros(var_count), np.zeros(var_count)
    for i in range(var_count):
        norm[i], p[i] = scipy_stats.normaltest(
            data[:, i], nan_policy=nan_policy
        )
    return norm, p


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
        'omit': remove NaN values
    """
    arr = remove_nan(arr) if const_policy == "omit" else arr

    try:
        box_cox_data, _ = scipy_stats.boxcox(arr, alpha=alpha, lmbda=lmbda)
    except ValueError as e:
        if const_policy == "propagate":
            return np.array([np.nan] * arr.shape[0])
        raise e

    return box_cox_data


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


def avg_moving_range(arr, nan_policy="omit"):
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
    np.ndarray
        Average moving range
    """
    arr_no_nan = remove_nan(arr)
    if len(arr_no_nan) != len(arr):
        if nan_policy == "raise":
            msg = "NaN values are not supported for avg_moving_range"
            raise NotImplementedError(msg)
        if nan_policy == "propagate":
            return np.nan
    if len(arr_no_nan) == 0:
        return np.nan
    return np.mean(np.absolute(np.diff(arr_no_nan)))


class ControlChartData:
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
        self.std = std
        self.ucl_limit = ucl_limit
        self.lcl_limit = lcl_limit
        self.x = x

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
        """Center line of charting data (i.e., mean value)"""
        data = remove_nan(self.y)
        if len(data):
            return np.mean(data)
        return np.nan

    @property
    def avg_moving_range(self):
        """Avg moving range based on 2 consecutive points"""
        return avg_moving_range(self.y, nan_policy="omit")

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


class HotellingT2Data:
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(self, data, alpha=0.05):
        """Initialize the Hotelling T^2 class

        Parameters
        ----------
        data : np.ndarray
            A 2-D array of data to perform multivariate analysis.
            (e.g., DVHAStats.data)
        alpha : float
            The significance level used to calculate the
            upper control limit (UCL)
        """

        self.data = data
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
        """Number of observations in data"""
        return self.data.shape[0]

    @property
    def variable_count(self):
        """Number of variables in data"""
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
        """Center line (median value of beta distribution)"""
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
        return ((N - 1) ** 2 / N) * scipy_stats.beta.ppf(x, a, b)


class PCA(sklearnPCA):
    """Hotelling's t-squared statistic for multivariate hypothesis testing"""

    def __init__(self, X, n_components=0.95, transform=True, **kwargs):
        """Initialize PCA and perform fit. Inherits sklearn.decomposition.PCA

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
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
        sklearnPCA.__init__(self, n_components=n_components, **kwargs)

        if transform:
            self.fit_transform(self.X)
        else:
            self.fit(self.X)

    @property
    def feature_map_data(self):
        return self.components_


class CorrelationMatrixData:
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
