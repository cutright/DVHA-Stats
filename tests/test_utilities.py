#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# test_utilities.py
"""unittest cases for utilities."""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


import unittest
from os.path import join
import numpy as np
from numpy.testing import assert_array_equal
from dvhastats import utilities
from dateutil.parser import parse as date_parser
from dateutil.parser._parser import ParserError

basedata_dir = join("tests", "testdata")
example_data = join(basedata_dir, "multivariate_data_small.csv")
example_data_nh = join(basedata_dir, "multivariate_data_small_no-header.csv")
example_narrow_data = join(basedata_dir, "narrow_data.csv")


class TestUtilities(unittest.TestCase):
    """Unit tests for Utilities."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        self.data_path = example_data
        self.data_path_nh = example_data_nh

        data = [
            [np.nan, 48.1, 48.3, 65.1, 47.1, 49.9, 49.5, 48.9, 35.5, 44.5],
            [51.9, 44.3, 44.5, 58.7, 41.1, 43.9, 43.9, 44.5, 31.1, 40.5],
            [48.5, 38.5, 37.1, 53.9, 27.1, 40.7, 34.5, 41.7, 28.7, 37.7],
            [33.9, 21.5, 20.1, 48.7, 12.1, 38.5, 13.9, 33.7, 25.3, 28.1],
            [19.1, 10.9, 9.9, 42.3, 4.3, 36.5, 6.1, 16.9, 13.9, 13.3],
            [12.7, 4.7, 3.9, 31.3, 3.1, 29.1, 3.7, 4.9, 6.9, 4.7],
        ]
        keys = ["V%s" % (i + 1) for i in range(6)]

        self.expected_dict = {key: data[i] for i, key in enumerate(keys)}
        self.expected_dict_nh = {i: row for i, row in enumerate(data)}
        self.expected_arr = np.array(data).T
        self.expected_var_names = keys

    def test_csv_to_dict(self):
        """Test csv_to_dict"""
        data = utilities.csv_to_dict(self.data_path, dtype=float)
        self.assertEqual(data, self.expected_dict)

    def test_dict_to_array(self):
        """Test dict_to_array"""
        dict_data = utilities.csv_to_dict(self.data_path, dtype=float)
        arr = utilities.dict_to_array(dict_data)
        assert_array_equal(arr["data"], self.expected_arr)
        self.assertEqual(arr["var_names"], self.expected_var_names)

    def test_no_header_row(self):
        """Test import with no header row"""
        data = utilities.csv_to_dict(
            self.data_path_nh, dtype=float, header_row=False
        )
        self.assertEqual(data, self.expected_dict_nh)

    def test_apply_dtype(self):
        """Test the apply_dtype function"""
        test = utilities.apply_dtype("2.5", float)
        self.assertEqual(test, 2.5)
        test = utilities.apply_dtype(2.5, str)
        self.assertEqual(test, "2.5")
        test = utilities.apply_dtype(2.5, None)
        self.assertEqual(test, 2.5)

    def test_import_data(self):
        """Test the generalized import data function"""
        # File test
        data, var_names = utilities.import_data(self.data_path)
        self.assertEqual(var_names, self.expected_var_names)
        assert_array_equal(data, self.expected_arr)

        # dict test
        data_dict = utilities.csv_to_dict(self.data_path, dtype=float)
        data_arr, var_names = utilities.import_data(data_dict)
        self.assertEqual(var_names, self.expected_var_names)
        assert_array_equal(data_arr, self.expected_arr)

        # array test
        data_arr_2, var_names = utilities.import_data(data_arr)
        self.assertEqual(var_names, list(range(len(data_arr[0, :]))))
        assert_array_equal(data_arr_2, self.expected_arr)

        with self.assertRaises(NotImplementedError):
            utilities.import_data([0, 1])

    def test_get_sorted_indices(self):
        """Test the simple get sorted indices function"""
        data = [0, 3, 2]
        test = utilities.get_sorted_indices(data)
        self.assertEqual(test, [0, 2, 1])

    def test_sort_2d_array(self):
        """Test the 2D array sort using numpy"""
        # One column sort
        arr = np.copy(self.expected_arr)
        arr = utilities.sort_2d_array(arr, 0)
        expected_sort = np.array(
            [
                [35.5, 31.1, 28.7, 25.3, 13.9, 6.9],
                [44.5, 40.5, 37.7, 28.1, 13.3, 4.7],
                [47.1, 41.1, 27.1, 12.1, 4.3, 3.1],
                [48.1, 44.3, 38.5, 21.5, 10.9, 4.7],
                [48.3, 44.5, 37.1, 20.1, 9.9, 3.9],
                [48.9, 44.5, 41.7, 33.7, 16.9, 4.9],
                [49.5, 43.9, 34.5, 13.9, 6.1, 3.7],
                [49.9, 43.9, 40.7, 38.5, 36.5, 29.1],
                [65.1, 58.7, 53.9, 48.7, 42.3, 31.3],
                [np.nan, 51.9, 48.5, 33.9, 19.1, 12.7],
            ]
        )
        assert_array_equal(arr, expected_sort)

        # Two column sort
        arr = np.copy(self.expected_arr)
        arr = utilities.sort_2d_array(arr, [1, 0])
        expected_sort = np.array(
            [
                [35.5, 31.1, 28.7, 25.3, 13.9, 6.9],
                [44.5, 40.5, 37.7, 28.1, 13.3, 4.7],
                [47.1, 41.1, 27.1, 12.1, 4.3, 3.1],
                [49.5, 43.9, 34.5, 13.9, 6.1, 3.7],
                [49.9, 43.9, 40.7, 38.5, 36.5, 29.1],
                [48.1, 44.3, 38.5, 21.5, 10.9, 4.7],
                [48.3, 44.5, 37.1, 20.1, 9.9, 3.9],
                [48.9, 44.5, 41.7, 33.7, 16.9, 4.9],
                [np.nan, 51.9, 48.5, 33.9, 19.1, 12.7],
                [65.1, 58.7, 53.9, 48.7, 42.3, 31.3],
            ]
        )
        assert_array_equal(arr, expected_sort)

        # Verify mode check
        with self.assertRaises(NotImplementedError):
            utilities.sort_2d_array(arr, 0, mode="test")

    def test_str_arr_to_date_arr(self):
        """Test str_arr_to_date_arr (str to datetime)"""
        # simple test
        data_str = ["1/2/2000", "5/1/2005"]
        data_dt = [date_parser(date) for date in data_str]
        test = utilities.str_arr_to_date_arr(data_str)
        self.assertEqual(test, data_dt)

        # unparsable str
        data_str.append("1/4/2004a")
        with self.assertRaises(ParserError):
            utilities.str_arr_to_date_arr(data_str)

        # force parsing
        test = utilities.str_arr_to_date_arr(data_str, force=True)
        expected = data_dt + [data_str[-1]]
        assert_array_equal(test, expected)

    def test_widen_data(self):
        """Test widen data script"""
        data_dict = utilities.csv_to_dict(example_narrow_data)
        kwargs = {
            "uid_columns": ["patient", "plan", "field id"],
            "x_data_cols": ["DD(%)", "DTA(mm)", "Threshold(%)"],
            "y_data_col": "Gamma Pass Rate(%)",
            "date_col": "date",
            "dtype": float,
        }
        ds = utilities.widen_data(data_dict, **kwargs)

        expected = {
            "uid": ["ANON1234Plan_name3", "ANON1234Plan_name4"],
            "date": ["6/13/2019 7:27", "6/13/2019 7:27"],
            "2.0 && 3.0 && 10.0": [np.nan, 99.99476863],
            "2.0 && 3.0 && 5.0": [99.88772435, 99.99533258],
            "3.0 && 2.0 && 10.0": [99.94708217, 99.99941874],
            "3.0 && 3.0 && 10.0": [99.97706894, 100],
            "3.0 && 3.0 && 5.0": [99.97934552, 100],
        }

        for key, exp_value in expected.items():
            assert_array_equal(ds[key], exp_value)

        # No date test
        kwargs_no_date = {key: value for key, value in kwargs.items()}
        kwargs_no_date["date_col"] = None
        ds_2 = utilities.widen_data(data_dict, **kwargs_no_date)
        for key, ds_2_value in ds_2.items():
            assert_array_equal(ds_2_value, expected[key])

        # test column length check
        data_dict_2 = utilities.csv_to_dict(example_narrow_data)
        data_dict_2[list(data_dict_2)[0]].append("test")
        with self.assertRaises(NotImplementedError):
            utilities.widen_data(data_dict_2, **kwargs)

        # test policy check
        with self.assertRaises(NotImplementedError):
            utilities.widen_data(data_dict, multi_val_policy="test", **kwargs)

        ds = utilities.widen_data(data_dict, multi_val_policy="last", **kwargs)
        index = ds["uid"].index("ANON1234Plan_name4")
        self.assertEqual(ds["2.0 && 3.0 && 10.0"][index], 50)

        ds = utilities.widen_data(data_dict, multi_val_policy="min", **kwargs)
        index = ds["uid"].index("ANON1234Plan_name4")
        self.assertEqual(ds["2.0 && 3.0 && 10.0"][index], 50)

        ds = utilities.widen_data(data_dict, multi_val_policy="max", **kwargs)
        index = ds["uid"].index("ANON1234Plan_name4")
        self.assertEqual(
            ds["2.0 && 3.0 && 10.0"][index],
            expected["2.0 && 3.0 && 10.0"][index],
        )

        ds = utilities.widen_data(
            data_dict, remove_partial_columns=True, **kwargs
        )
        self.assertTrue("2.0 && 3.0 && 10.0" not in ds.keys())

    def test_is_numeric(self):
        self.assertTrue(utilities.is_numeric(3))
        self.assertFalse(utilities.is_numeric('a'))


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
