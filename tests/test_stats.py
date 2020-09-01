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
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dvhastats import stats
import warnings

basedata_dir = join("tests", "testdata")
example_data = join(basedata_dir, "multivariate_data_small.csv")


class TestStats(unittest.TestCase):
    """Unit tests for Stats."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        self.data_path = example_data

        self.expected_dict = {
            "V1": [56.5, 48.1, 48.3, 65.1, 47.1, 49.9, 49.5, 48.9, 35.5, 44.5],
            "V2": [51.9, 44.3, 44.5, 58.7, 41.1, 43.9, 43.9, 44.5, 31.1, 40.5],
            "V3": [48.5, 38.5, 37.1, 53.9, 27.1, 40.7, 34.5, 41.7, 28.7, 37.7],
            "V4": [33.9, 21.5, 20.1, 48.7, 12.1, 38.5, 13.9, 33.7, 25.3, 28.1],
            "V5": [19.1, 10.9, 9.9, 42.3, 4.3, 36.5, 6.1, 16.9, 13.9, 13.3],
            "V6": [12.7, 4.7, 3.9, 31.3, 3.1, 29.1, 3.7, 4.9, 6.9, 4.7],
        }
        self.expected_var_names = sorted(list(self.expected_dict))
        self.expected_arr = np.array(
            [
                [56.5, 48.1, 48.3, 65.1, 47.1, 49.9, 49.5, 48.9, 35.5, 44.5],
                [51.9, 44.3, 44.5, 58.7, 41.1, 43.9, 43.9, 44.5, 31.1, 40.5],
                [48.5, 38.5, 37.1, 53.9, 27.1, 40.7, 34.5, 41.7, 28.7, 37.7],
                [33.9, 21.5, 20.1, 48.7, 12.1, 38.5, 13.9, 33.7, 25.3, 28.1],
                [19.1, 10.9, 9.9, 42.3, 4.3, 36.5, 6.1, 16.9, 13.9, 13.3],
                [12.7, 4.7, 3.9, 31.3, 3.1, 29.1, 3.7, 4.9, 6.9, 4.7],
            ]
        ).T
        self.stats_obj = stats.DVHAStats(self.data_path)

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

    def test_get_index_by_var_name(self):
        self.assertEqual(0, self.stats_obj.get_index_by_var_name("V1"))
        with self.assertRaises(AttributeError):
            self.stats_obj.get_index_by_var_name("test")

    def test_get_data_by_var_name(self):
        expected = self.expected_arr[:, 0]
        data = self.stats_obj.get_data_by_var_name("V1")
        assert_array_equal(data, expected)

    def test_observations(self):
        self.assertEqual(self.stats_obj.observations, 10)

    def test_variable_count(self):
        self.assertEqual(self.stats_obj.variable_count, 6)

    def test_pearson_r_matrix(self):
        exp_r = np.array(
            [
                [
                    1.0,
                    0.99264939,
                    0.86011723,
                    0.58662221,
                    0.61300906,
                    0.62923282,
                ],
                [
                    0.99264939,
                    1.0,
                    0.88851061,
                    0.58486366,
                    0.58064696,
                    0.5791157,
                ],
                [
                    0.86011723,
                    0.88851061,
                    1.0,
                    0.83404009,
                    0.74483928,
                    0.66846345,
                ],
                [
                    0.58662221,
                    0.58486366,
                    0.83404009,
                    1.0,
                    0.93433324,
                    0.83444095,
                ],
                [
                    0.61300906,
                    0.58064696,
                    0.74483928,
                    0.93433324,
                    1.0,
                    0.97058164,
                ],
                [
                    0.62923282,
                    0.5791157,
                    0.66846345,
                    0.83444095,
                    0.97058164,
                    1.0,
                ],
            ]
        )
        exp_p = np.array(
            [
                [
                    0.00000000e00,
                    1.26600528e-08,
                    1.40996093e-03,
                    7.46506281e-02,
                    5.94985157e-02,
                    5.12738548e-02,
                ],
                [
                    1.26600528e-08,
                    0.00000000e00,
                    5.89647474e-04,
                    7.57410908e-02,
                    7.83977572e-02,
                    7.93772216e-02,
                ],
                [
                    1.40996093e-03,
                    5.89647474e-04,
                    0.00000000e00,
                    2.70253622e-03,
                    1.34485527e-02,
                    3.45956064e-02,
                ],
                [
                    7.46506281e-02,
                    7.57410908e-02,
                    2.70253622e-03,
                    0.00000000e00,
                    7.51138400e-05,
                    2.67788998e-03,
                ],
                [
                    5.94985157e-02,
                    7.83977572e-02,
                    1.34485527e-02,
                    7.51138400e-05,
                    0.00000000e00,
                    3.16254737e-06,
                ],
                [
                    5.12738548e-02,
                    7.93772216e-02,
                    3.45956064e-02,
                    2.67788998e-03,
                    3.16254737e-06,
                    0.00000000e00,
                ],
            ]
        )
        r, p = self.stats_obj.pearson_r_matrix
        assert_array_almost_equal(r, exp_r)
        assert_array_almost_equal(p, exp_p)

    def test_normality(self):
        expected_norm = np.array(
            [
                2.85071988,
                2.16842411,
                0.48490695,
                0.36635239,
                3.99716349,
                5.49795211,
            ]
        )
        expected_p = np.array(
            [
                0.24042191,
                0.33816814,
                0.78470025,
                0.83262144,
                0.13552736,
                0.06399335,
            ]
        )
        warnings.filterwarnings("ignore")
        norm, p = self.stats_obj.normality
        warnings.filterwarnings("default")
        assert_array_almost_equal(norm, expected_norm)
        assert_array_almost_equal(p, expected_p)

    def test_univariate_control_chart(self):
        ucc = self.stats_obj.univariate_control_charts()
        self.assertEqual(round(ucc[0].center_line, 3), 49.340)
        lcl, ucl = ucc[0].control_limits
        self.assertEqual(round(lcl, 3), 28.773)
        self.assertEqual(round(ucl, 3), 69.907)
        self.assertEqual(len(ucc[0].out_of_control), 0)

    def test_hotelling_t2(self):
        ht2 = self.stats_obj.hotelling_t2()
        self.assertEqual(round(ht2.center_line, 3), 5.614)
        lcl, ucl = ht2.control_limits
        self.assertEqual(round(lcl, 3), 0)
        self.assertEqual(round(ucl, 3), 7.834)
        self.assertEqual(len(ht2.out_of_control), 0)


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
