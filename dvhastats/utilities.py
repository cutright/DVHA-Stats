#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# utilities.py
"""Common functions for the DVHA-Stats."""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats

import numpy as np


def apply_dtype(value, dtype):
    if dtype is None:
        return value
    try:
        value = dtype(value)
    except ValueError:
        value = np.nan
    return value


def csv_to_dict(csv_file_path, delimiter=",", dtype=None, header_row=True):
    """Read in a csv file, return data as a dictionary

    Parameters
    ----------
    csv_file_path : str
        File path to the CSV file to be processed.
    delimiter : str
        Specify the delimiter used in the csv file (default = ',')
    dtype : None, Type, optional
        Optionally force values to a type (e.g., float, int, str, etc.).
    header_row : bool, optional
        If True, the first row is interpreted as column keys, otherwise row
        indices will be used

    Returns
    -------
    dict
        CSV data as a dict, using the first row values as keys
    """

    with open(csv_file_path) as fp:

        # Read first row, determine column keys
        first_row = fp.readline().strip().split(delimiter)
        if header_row:
            keys = first_row
            data = {key: [] for key in keys}
        else:
            keys = list(range(len(first_row)))
            data = {key: [apply_dtype(first_row[key], dtype)] for key in keys}

        # Iterate through remaining rows, append values to data
        for r, line in enumerate(fp):
            row = line.strip().split(",")
            for c, value in enumerate(row):
                data[keys[c]].append(apply_dtype(value, dtype))

    return data


def dict_to_array(data, key_order=None):
    """Convert a dict of data to a numpy array

    Parameters
    ----------
    data : dict
        File path to the CSV file to be processed.
    key_order : None, list of str
        Optionally the order of columns

    Returns
    -------
    dict
        A dictionary with keys of 'data' and 'columns', pointing to a
        numpy array and list of str, respectively
    """
    var_names = key_order if key_order is not None else list(data.keys())
    arr_data = [data[key] for key in var_names]
    return {"data": np.asarray(arr_data).T, "var_names": var_names}


def moving_avg(y, avg_len, x=None, weight=None):
    """Calculate the moving (rolling) average of a set of data

    Parameters
    ----------
    y : np.ndarray, list
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
