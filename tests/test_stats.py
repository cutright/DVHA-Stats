#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# test_stats.py
"""unittest cases for stats."""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


import unittest
from os.path import join
import numpy as np
from numpy.testing import assert_array_equal
from dvhastats import stats

basedata_dir = join("tests", "testdata")
example_data = join(basedata_dir, "multivariate_data_small.csv")


class TestStats(unittest.TestCase):
    """Unit tests for Stats."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        self.data_path = example_data

        self.expected_dict = {
            "V1": [56.5, 48.1, 48.3, 65.1, 47.1],
            "V2": [51.9, 44.3, 44.5, 58.7, 41.1],
            "V3": [48.5, 38.5, 37.1, 53.9, 27.1],
            "V4": [33.9, 21.5, 20.1, 48.7, 12.1],
            "V5": [19.1, 10.9, 9.9, 42.3, 4.3],
            "V6": [12.7, 4.7, 3.9, 31.3, 3.1],
        }
        self.expected_var_names = sorted(list(self.expected_dict))
        self.expected_arr = np.array(
            [
                [56.5, 48.1, 48.3, 65.1, 47.1],
                [51.9, 44.3, 44.5, 58.7, 41.1],
                [48.5, 38.5, 37.1, 53.9, 27.1],
                [33.9, 21.5, 20.1, 48.7, 12.1],
                [19.1, 10.9, 9.9, 42.3, 4.3],
                [12.7, 4.7, 3.9, 31.3, 3.1],
            ]
        ).T

    def test_arr_import(self):
        """Test array importing"""
        stats_obj = stats.DVHAStats(
            self.expected_arr, var_names=self.expected_var_names
        )
        assert_array_equal(stats_obj.data, self.expected_arr)
        self.assertEqual(stats_obj.var_names, self.expected_var_names)

    def test_dict_import(self):
        """Test dict importing"""
        stats_obj = stats.DVHAStats(self.expected_dict)
        assert_array_equal(stats_obj.data, self.expected_arr)
        self.assertEqual(stats_obj.var_names, self.expected_var_names)

    def test_csv_import(self):
        """Test csv importing"""
        stats_obj = stats.DVHAStats(self.data_path)
        assert_array_equal(stats_obj.data, self.expected_arr)
        self.assertEqual(stats_obj.var_names, self.expected_var_names)


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
