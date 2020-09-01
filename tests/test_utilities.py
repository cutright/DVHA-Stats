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
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dvhastats import utilities

basedata_dir = join("tests", "testdata")
example_data = join(basedata_dir, "multivariate_data_small.csv")
example_data_nh = join(basedata_dir, "multivariate_data_small_no-header.csv")


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

    def test_moving_avg(self):
        avg_len = 5
        y = np.array([4, 23, 1, 4, -18, 2.2, 100, 34, 9, 8])
        x = np.linspace(1, len(y), len(y))
        x_avg, y_avg = utilities.moving_avg(y, avg_len)
        assert_array_equal(x_avg, x[avg_len - 1 : :])
        y_avg_exp = np.array([2.8, 2.44, 17.84, 24.44, 25.44, 30.64])
        assert_array_almost_equal(y_avg, y_avg_exp)

        avg_len = 3
        x += 10
        w = np.array([1, 2, 1, 1, 5, 2, 1, 1, 1, 1])
        x_avg, y_avg = utilities.moving_avg(y, avg_len, x=x, weight=w)
        assert_array_equal(x_avg, x[avg_len - 1 : :])
        y_avg_exp = np.array(
            [5.5, 5.5, 0.466667, 0.5, 32.5, 45.033333, 47.666667, 17.0]
        )
        assert_array_almost_equal(y_avg, y_avg_exp)


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
