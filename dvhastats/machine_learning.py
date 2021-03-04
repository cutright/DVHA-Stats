#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# machine_learning.py
"""Machine Learning class objects"""
#
# Copyright (c) 2021 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats

import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


ALGORITHMS = {
    "Random Forest": {
        "regressor": RandomForestRegressor,
        "classifier": RandomForestClassifier,
    },
    "Support Vector Machine": {
        "regressor": SVR,
        "classifier": SVC,
    },
    "Gradient Boosting": {
        "regressor": GradientBoostingRegressor,
        "classifier": GradientBoostingClassifier,
    },
    "Decision Tree": {
        "regressor": DecisionTreeRegressor,
        "classifier": DecisionTreeClassifier,
    },
    "Multilayer Perceptron": {
        "regressor": MLPRegressor,
        "classifier": MLPClassifier,
    },
}


# TODO: re-enable in .coveragerc when ready
class MachineLearning:
    """Generic Machine Learning class based on scikit-learn

    Parameters
    ----------
    alg : str or scikit-learn predictor
        Built-in options include 'Random Forest', 'Support Vector Machine',
        'Gradient Boosting', 'Decision Tree', 'Multilayer Perceptron'. Must
        also specify ``alg_type`` for built-in options. Alternatively,
        provide any other scikit-learn prediction class.
    alg_type : str, optional
        Required if ``alg`` is a string. either 'regressor' or 'classifier'
    **input_parameters : optional kwargs
        All other keyword arguments will be passed into the scikit-learn
        predictor object. Please refer to scikit-learn documentation.
    """

    def __init__(
        self,
        X,
        y,
        alg,
        alg_type="regressor",
        test_size=0.25,
        train_size=None,
        random_state=None,
        shuffle=True,
        **input_parameters
    ):
        self.X = X
        self.y = y
        self.alg_type = alg_type
        if isinstance(alg, str):
            if alg_type in ["regressor", "classifier"]:
                self.predictor_class = ALGORITHMS[alg_type]
            else:
                msg = "'alg_type' must be either 'regressor' or 'classifier'"
                raise NotImplementedError(msg)
        else:
            self.predictor_class = alg

        self.data_split_parameters = {
            "test_size": test_size,
            "train_size": train_size,
            "random_state": random_state,
            "shuffle": shuffle,
        }

        self.input_parameters = input_parameters

        self.model = self.predictor_class(**self.input_parameters)

    def __call__(self):
        """Do training

        Returns
        -------
        PlotData
            A trained machine learning model
        """
        return PlotData(
            self.X, self.y, self.model, **self.data_split_parameters
        )

    def grid_search_tuning(self, param_grid, **kwargs):
        """Tune model parameters with a GridSearchCV

        Parameters
        ----------
        param_grid :
        """
        search = GridSearchCV(self.model, param_grid, **kwargs)
        search.fit(self.X, self.y)
        return search

    def random_search_tuning(self, param_grid, n_iter=100, **kwargs):
        """Tune model parameters with a RandomizedSearchCV"""
        search = RandomizedSearchCV(self.model, param_grid, n_iter, **kwargs)
        search.fit(self.X, self.y)
        return search


class PlotData:
    """Perform training and get data for plotting

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values
    model : scikit-learn predictor class object
        scikit-learn predictor class object
    """

    def __init__(
        self,
        X,
        y,
        model,
        do_training=True,
        test_size=0.25,
        train_size=None,
        random_state=None,
        shuffle=True,
    ):
        self.model = model
        self.split_args = {
            "test_size": test_size,
            "train_size": train_size,
            "random_state": random_state,
            "shuffle": shuffle,
        }

        indices = list(range(len(y)))

        # split the data for training and testing
        split_data = train_test_split(X, indices, **self.split_args)
        self.X = {"data": X, "train": split_data[0], "test": split_data[1]}
        self.indices = {
            "data": indices,
            "train": split_data[2],
            "test": split_data[3],
        }
        self.y = {
            "data": y,
            "train": [y[i] for i in split_data[2]],
            "test": [y[i] for i in split_data[3]],
        }
        self.x = {
            key: [i + 1 for i in range(len(data))]
            for key, data in self.y.items()
        }

        # Train model, then calculate predictions, residuals, and mse
        if do_training:
            self.model.fit(self.X["train"], self.y["train"])
        self.predictions = {
            key: self.get_prediction(key) for key in self.y.keys()
        }
        self.residuals = {key: self.get_residual(key) for key in self.y.keys()}
        self.mse = {key: self.get_mse(key) for key in self.y.keys()}
        self.accuracy = {key: self.get_accuracy(key) for key in self.y.keys()}

    def get_prediction(self, key):
        """Calculate predicted values

        Parameters
        ----------
        key : str
            Either 'train' or 'test'

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            predictions
        """
        return self.model.predict(self.X[key])

    def get_mse(self, key):
        """Calculate mean square error (MSE)

        Parameters
        ----------
        key : str
            Either 'train' or 'test'

        Returns
        -------
        float
            Mean Square Error
        """
        return np.mean(
            np.square(np.subtract(self.predictions[key], self.y[key]))
        )

    def get_residual(self, key):
        """Calculate residuals (for regressions only)

        Parameters
        ----------
        key : str
            Either 'train' or 'test'

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            y - prediction
        """
        return np.subtract(self.y[key], self.model.predict(self.X[key]))

    def get_accuracy(self, key):
        """Calculate accuracy (for classifiers only)

        Parameters
        ----------
        key : str
            Either 'train' or 'test'

        Returns
        -------
        float
            Correct prediction divided by length of y
        """
        return np.count_nonzero(
            np.subtract(self.predictions[key], self.y[key]) == 0
        ) / len(self.y[key])

    @property
    def feature_importances(self):
        """Get the feature importances (if available)

        Returns
        -------
        ndarray of shape (n_features,)
            Feature importances
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
