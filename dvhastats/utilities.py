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
from dateutil.parser import parse as date_parser
import csv


def apply_dtype(value, dtype):
    """Convert value with the provided data type

    Parameters
    ----------
    value : any
        Value to be converted
    dtype : function, None
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
    dtype : callable, type, optional
        Optionally force values to a type (e.g., float, int, str, etc.).
    header_row : bool, optional
        If True, the first row is interpreted as column keys, otherwise row
        indices will be used

    Returns
    -------
    dict
        CSV data as a dict, using the first row values as keys
    """

    with open(csv_file_path, "r") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        if header_row:
            first_row = next(reader)
            keys = [key.strip() for key in first_row]
            data = list(reader)
        else:
            data = list(reader)
            keys = list(range(len(data[0])))

    data_dict = {key: [] for key in keys}
    for row in data:
        for c, value in enumerate(row):
            data_dict[keys[c]].append(apply_dtype(value, dtype))

    return data_dict


def dict_to_array(data, key_order=None):
    """Convert a dict of data to a numpy array

    Parameters
    ----------
    data : dict
        Dictionary of data to be converted to np.array.
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
    if isinstance(data, str) and isfile(data):
        if splitext(data)[1] == ".csv":
            data = dict_to_array(csv_to_dict(data, dtype=float))
            return data["data"], data["var_names"]

    msg = "Invalid data provided - must be a numpy array, dict, or .csv file"
    raise NotImplementedError(msg)


def widen_data(
    data_dict,
    uid_columns,
    x_data_cols,
    y_data_col,
    date_col=None,
    sort_by_date=True,
    remove_partial_columns=False,
    multi_val_policy="first",
    dtype=None,
    date_parser_kwargs=None,
):
    """Convert a narrow data dictionary into wide format (i.e., from one row
    per dependent value to one row per observation)

    Parameters
    ----------
    data_dict : dict
        Data to be converted. The length of each array must be uniform.
    uid_columns : list
        Keys of data_dict used to create an observation uid
    x_data_cols : list
        Keys of columns representing independent data
    y_data_col : int, str
        Key of data_dict representing dependent data
    date_col : int, str, optional
        Key of date column
    sort_by_date : bool, optional
        Sort output by date (date_col required)
    remove_partial_columns : bool, optional
        If true, any columns that have a blank row will be removed
    multi_val_policy : str
        Either 'first', 'last', 'min', 'max'. If multiple values are found for
        a particular combination of x_data_cols, one value will be selected
        based on this policy.
    dtype : function
        python reserved types, e.g., int, float, str, etc. However, dtype
        could be any callable that raises a ValueError on failure.
    date_parser_kwargs : dict, optional
        Keyword arguments to be passed into dateutil.parser.parse

    Returns
    ----------
    dict
        data_dict reformatted to one row per UID
    """

    data_lengths = [len(col) for col in data_dict.values()]
    if len(set(data_lengths)) != 1:
        msg = "Each column of data_dict must be of the same length"
        raise NotImplementedError(msg)

    if multi_val_policy not in {"first", "last", "min", "max"}:
        msg = "multi_val_policy must be in 'first', 'last', 'min', or 'max'"
        raise NotImplementedError(msg)

    data = {}
    for row in range(len(data_dict[y_data_col])):
        uid = "".join([str(data_dict[col][row]) for col in uid_columns])

        if uid not in list(data):
            data[uid] = {}

        vals = [data_dict[col][row] for col in x_data_cols]
        vals = [float(v) if is_numeric(v) else v for v in vals]
        params = " && ".join([str(v) for v in vals])

        date = 0 if date_col is None else data_dict[date_col][row]
        if date not in data[uid].keys():
            data[uid][date] = {}

        if params not in list(data[uid][date]):
            data[uid][date][params] = []

        data[uid][date][params].append(data_dict[y_data_col][row])

    x_variables = []
    for results in data.values():
        for date_results in results.values():
            for param in date_results.keys():
                if param not in {"uid", "date"}:
                    x_variables.append(param)
    x_variables = sorted(list(set(x_variables)))

    keys = ["uid", "date"] + x_variables
    wide_data = {key: [] for key in keys}
    partial_cols = []
    for uid, date_data in data.items():
        for date, param_data in date_data.items():
            wide_data["uid"].append(uid)
            wide_data["date"].append(date)
            for x in x_variables:
                values = param_data.get(x)
                if values is None:
                    if remove_partial_columns:
                        partial_cols.append(x)
                    values = [""]

                if dtype is not None:
                    values = [apply_dtype(v, dtype) for v in values]

                value = values[0]
                if len(values) > 1:
                    print(
                        "WARNING: Multiple values found for uid: %s, date: "
                        "%s, param: %s. Only the %s value is included in "
                        "widen_data output." % (uid, date, x, multi_val_policy)
                    )
                    if multi_val_policy == "last":
                        value = values[-1]
                    elif multi_val_policy in {"min", "max"}:
                        value = {"min": min, "max": max}[multi_val_policy](
                            values
                        )

                wide_data[x].append(value)

    if remove_partial_columns:
        partial_cols = set(partial_cols)
        if len(partial_cols):
            for col in partial_cols:
                wide_data.pop(col)
                x_variables.pop(x_variables.index(col))

    if date_col is None:
        wide_data.pop("date")
    elif sort_by_date:
        kwargs = {} if date_parser_kwargs is None else date_parser_kwargs
        dates = str_arr_to_date_arr(wide_data["date"], kwargs)
        sorted_indices = get_sorted_indices(dates)
        final_data = {key: [] for key in wide_data.keys()}
        for row in range(len(wide_data[x_variables[0]])):
            final_data["uid"].append(wide_data["uid"][sorted_indices[row]])
            final_data["date"].append(wide_data["date"][sorted_indices[row]])
            for x in x_variables:
                final_data[x].append(wide_data[x][sorted_indices[row]])
        return final_data

    return wide_data


def get_sorted_indices(list_data):
    """Get original indices of a list after sorting

    Parameters
    ----------
    list_data : list
        Any python sortable list

    Returns
    ----------
    list
        list_data indices of sorted(list_data)
    """
    return [i[0] for i in sorted(enumerate(list_data), key=lambda x: x[1])]


def sort_2d_array(arr, index, mode="col"):
    """Sort a 2-D numpy array

    Parameters
    ----------
    arr : np.ndarray
        Input 2-D array to be sorted
    index : int, list
        Index of column or row to sort arr.  If list, will sort by each index
        in the order provided.
    mode : str
        Either 'col' or 'row'
    """
    if not isinstance(index, list):
        index = [index]

    if mode not in {"col", "row"}:
        msg = (
            "Unsupported sort_2d_array mode, "
            "must be either 'col' or 'row' - got %s" % mode
        )
        raise NotImplementedError(msg)

    sort_by = arr[:, index[-1]] if mode == "col" else arr[index[-1], :]
    arr = arr[sort_by.argsort()]
    for i in index[0:-1][::-1]:
        sort_by = arr[:, i] if mode == "col" else arr[i, :]
        arr = arr[sort_by.argsort(kind="mergesort")]
    return arr


def str_arr_to_date_arr(arr, date_parser_kwargs=None, force=False):
    """Convert an array of datetime strings to a list of datetime objects

    Parameters
    ----------
    arr : array-like
        Array of datetime strings compatible with dateutil.parser.parse
    date_parser_kwargs : dict, optional
        Keyword arguments to be passed into dateutil.parser.parse
    force : bool
        If true, failed parsings will result in original value. If false,
        dateutil.parser.parse's error will be raised on failures.

    Returns
    ----------
    list
        list of datetime objects
    """
    kwargs = {} if date_parser_kwargs is None else date_parser_kwargs
    dates = []
    for date_str in arr:
        try:
            date = date_parser(date_str, **kwargs)
        except Exception as e:
            if force:
                date = date_str
            else:
                raise e
        dates.append(date)
    return dates


def is_numeric(val):
    """Check if value is numeric (float or int)

    Parameters
    ----------
    val : any
        Any value

    Returns
    -------
    bool
        Returns true if float(val) doesn't raise a ValueError
    """
    try:
        float(val)
        return True
    except ValueError:
        return False
