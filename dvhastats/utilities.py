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
from os.path import isfile, splitext


def apply_dtype(value, dtype):
    """Convert value with the provided data type

    Parameters
    ----------
    value : any
        Value to be converted
    dtype : callable
        python reserved types, e.g., int, float, str, etc. However, dtype
        could be any callable that raises a ValueError on failure.

    Returns
    ----------
    any
        The return of dtype(value) or numpy.nan on ValueError
    """
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


def import_data(data, var_names=None):
    """Generalized data importer for np.ndarray, dict, and csv file

    Parameters
    ----------
    data : numpy.array, dict, str
        Input data (2-D) with N rows of observations and
        p columns of variables.  The CSV file must have a header row
        for column names.
    var_names : list of str, optional
        If data is a numpy array, optionally provide the column names.

    Returns
    ----------
    np.ndarray, list
        A tuple: data as an array and variable names as a list
    """
    if isinstance(data, np.ndarray):
        var_names = (
            var_names if var_names is not None else list(range(data.shape[1]))
        )
        return data, var_names
    if isinstance(data, dict):
        data = dict_to_array(data)
        return data["data"], data["var_names"]
    if isfile(data):
        if splitext(data)[1] == ".csv":
            data = dict_to_array(csv_to_dict(data, dtype=float))
            return data["data"], data["var_names"]

    msg = "Invalid data provided - must be a numpy array, dict, or .csv file"
    raise NotImplementedError(msg)
