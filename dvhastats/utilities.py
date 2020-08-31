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
            data = {key: [first_row[key]] for key in keys}

        # Iterate through remaining rows, append values to data
        for r, line in enumerate(fp):
            row = line.strip().split(",")
            for c, value in enumerate(row):
                if dtype is not None:
                    try:
                        value = dtype(value)
                    except ValueError:
                        value = np.nan
                data[keys[c]].append(value)

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
