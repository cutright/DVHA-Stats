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
import matplotlib
from copy import deepcopy

matplotlib.use("Template")

basedata_dir = join("tests", "testdata")
example_data = join(basedata_dir, "multivariate_data_small.csv")
example_data_full = join(basedata_dir, "multivariate_data.csv")


class TestStats(unittest.TestCase):
    """Unit tests for Stats."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        self.data_path = example_data
        self.data_path_full = example_data_full

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
        self.expected_dict_no_nan = deepcopy(self.expected_dict)
        self.expected_dict_no_nan["V1"][0] = 56.5
        self.expected_arr = np.array(data).T
        self.expected_var_names = keys

        self.const_data = deepcopy(self.expected_arr)
        self.const_data[:, 0] = np.ones_like(self.expected_arr[:, 0])

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

    def test_invalid_data_import(self):
        with self.assertRaises(NotImplementedError):
            stats.DVHAStats("test")

    def test_get_index_by_var_name(self):
        """Test data column index look-up by variable name"""
        self.assertEqual(0, self.stats_obj.get_index_by_var_name("V1"))
        with self.assertRaises(AttributeError):
            self.stats_obj.get_index_by_var_name("test")
        self.assertEqual(0, self.stats_obj.get_index_by_var_name(0))

    def test_get_data_by_var_name(self):
        """Test get data of a given var_name"""
        expected = self.expected_arr[:, 0]
        data = self.stats_obj.get_data_by_var_name("V1")
        assert_array_equal(data, expected)

    def test_observations(self):
        """Test DVHAStats.observations property"""
        self.assertEqual(self.stats_obj.observations, 10)

    def test_variable_count(self):
        """Test DVHAStats.variable_count property"""
        self.assertEqual(self.stats_obj.variable_count, 6)

    def test_correlation_matrix_type_failure(self):
        """Check that error is raised if corr_type Spearman or Pearson"""
        with self.assertRaises(NotImplementedError):
            self.stats_obj.correlation_matrix(corr_type="test")

    def test_pearson_r_matrix(self):
        """Test Pearson-R matrix calculation"""
        exp_r = np.array(
            [
                [
                    1.0,
                    0.99242287,
                    0.84208524,
                    0.56410877,
                    0.63275247,
                    0.6431104,
                ],
                [
                    0.99242287,
                    1.0,
                    0.88851061,
                    0.58486366,
                    0.58064696,
                    0.5791157,
                ],
                [
                    0.84208524,
                    0.88851061,
                    1.0,
                    0.83404009,
                    0.74483928,
                    0.66846345,
                ],
                [
                    0.56410877,
                    0.58486366,
                    0.83404009,
                    1.0,
                    0.93433324,
                    0.83444095,
                ],
                [
                    0.63275247,
                    0.58064696,
                    0.74483928,
                    0.93433324,
                    1.0,
                    0.97058164,
                ],
                [
                    0.6431104,
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
                    1.23765714e-07,
                    4.39932805e-03,
                    1.13622389e-01,
                    6.74147444e-02,
                    6.16994726e-02,
                ],
                [
                    1.23765714e-07,
                    0.00000000e00,
                    5.89647474e-04,
                    7.57410908e-02,
                    7.83977572e-02,
                    7.93772216e-02,
                ],
                [
                    4.39932805e-03,
                    5.89647474e-04,
                    0.00000000e00,
                    2.70253622e-03,
                    1.34485527e-02,
                    3.45956064e-02,
                ],
                [
                    1.13622389e-01,
                    7.57410908e-02,
                    2.70253622e-03,
                    0.00000000e00,
                    7.51138400e-05,
                    2.67788998e-03,
                ],
                [
                    6.74147444e-02,
                    7.83977572e-02,
                    1.34485527e-02,
                    7.51138400e-05,
                    0.00000000e00,
                    3.16254737e-06,
                ],
                [
                    6.16994726e-02,
                    7.93772216e-02,
                    3.45956064e-02,
                    2.67788998e-03,
                    3.16254737e-06,
                    0.00000000e00,
                ],
            ]
        )
        corr_mat = self.stats_obj.correlation_matrix()
        assert_array_almost_equal(corr_mat.corr, exp_r)
        assert_array_almost_equal(corr_mat.p, exp_p)

    def test_normality(self):
        """Test normality calculation"""
        expected_norm = np.array(
            [
                5.560821,
                2.16842411,
                0.48490695,
                0.36635239,
                3.99716349,
                5.49795211,
            ]
        )
        expected_p = np.array(
            [
                0.062013,
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
        """Test univariate control chart creation and values"""
        ucc = self.stats_obj.univariate_control_charts()
        self.assertEqual(round(ucc[0].center_line, 3), 48.544)
        lcl, ucl = ucc[0].control_limits
        self.assertEqual(round(lcl, 3), 28.199)
        self.assertEqual(round(ucl, 3), 68.89)
        self.assertEqual(len(ucc[0].out_of_control), 0)
        str_rep = (
            "center_line: 48.544\ncontrol_limits: "
            "28.199, 68.890\nout_of_control: []"
        )
        self.assertEqual(str(ucc[0]), str_rep)
        self.assertEqual(repr(ucc[0]), str_rep)

    def test_univariate_control_chart_with_limits(self):
        """Test univariate control chart creation and values"""
        ucc = self.stats_obj.univariate_control_charts(
            lcl_limit=30, ucl_limit=50
        )
        lcl, ucl = ucc[0].control_limits
        self.assertEqual(round(lcl, 3), 30)
        self.assertEqual(round(ucl, 3), 50)
        self.assertEqual(len(ucc[0].out_of_control), 1)

    def test_univariate_control_chart_box_cox(self):
        """Test univariate control chart creation and values with Box-Cox"""
        stats_obj = stats.DVHAStats(self.data_path_full)
        ucc = stats_obj.univariate_control_charts(box_cox=True)
        self.assertEqual(round(ucc[0].center_line, 3), 1835.702)
        lcl, ucl = ucc[0].control_limits
        self.assertEqual(round(lcl, 3), 136.258)
        self.assertEqual(round(ucl, 3), 3535.147)
        self.assertEqual(len(ucc[0].out_of_control), 3)
        self.assertEqual(len(ucc[0].out_of_control_high), 1)
        self.assertEqual(len(ucc[0].out_of_control_low), 2)

    def test_hotelling_t2(self):
        """Test multivariate control chart creation and values"""
        ht2 = self.stats_obj.hotelling_t2()
        self.assertEqual(round(ht2.center_line, 3), 5.614)
        lcl, ucl = ht2.control_limits
        self.assertEqual(round(lcl, 3), 0)
        self.assertEqual(round(ucl, 3), 7.834)
        self.assertEqual(len(ht2.out_of_control), 0)
        str_rep = (
            "Q: [nan nan nan nan nan nan nan nan nan nan]\n"
            "center_line: 5.614\n"
            "control_limits: 0, 7.834\n"
            "out_of_control: []"
        )
        self.assertEqual(str(ht2), str_rep)
        self.assertEqual(repr(ht2), str_rep)

    def test_hotelling_t2_box_cox(self):
        """Test multivariate control chart creation and values"""
        stats_obj = stats.DVHAStats(self.data_path_full)
        ht2 = stats_obj.hotelling_t2(box_cox=True)
        self.assertEqual(round(ht2.center_line, 3), 5.375)
        lcl, ucl = ht2.control_limits
        self.assertEqual(round(lcl, 3), 0)
        self.assertEqual(round(ucl, 3), 13.555)
        self.assertEqual(len(ht2.out_of_control), 2)

    def test_multi_variable_regression(self):
        """Test Multi-Variable Linear Regression"""
        y = np.linspace(1, 10, 10)
        stats_obj = stats.DVHAStats(self.expected_dict_no_nan)
        mvr = stats_obj.linear_reg(y)
        self.assertEqual(round(mvr.y_intercept, 3), 2.983)
        slope = np.array(
            [
                3.80907501,
                -4.68155991,
                0.67410364,
                -0.32432575,
                0.9811169,
                -1.10516451,
            ]
        )
        assert_array_almost_equal(mvr.slope, slope)
        self.assertEqual(round(mvr.mse, 3), 1.353)
        self.assertEqual(round(mvr.r_sq, 3), 0.836)
        residuals = np.array(
            [
                -0.62566871,
                -1.28605435,
                0.47511697,
                0.40490283,
                -0.11474329,
                -0.13508221,
                0.34435215,
                -1.26302838,
                -0.76163533,
                2.96184032,
            ]
        )
        assert_array_almost_equal(mvr.residuals, residuals)
        self.assertEqual(round(mvr.f_stat, 3), 2.548)
        self.assertEqual(mvr.df_error, 3)
        self.assertEqual(mvr.df_model, 6)
        self.assertEqual(round(mvr.f_p_value, 3), 0.763)

        mvr2 = stats_obj.linear_reg(y, saved_reg=mvr)
        assert_array_equal(mvr.residuals, mvr2.residuals)

    def test_box_cox_by_index(self):
        """Test box-cox transformation by index and keyword"""
        data_by_index = self.stats_obj.box_cox_by_index(0)
        data_by_key = self.stats_obj.box_cox_by_index("V1")
        assert_array_equal(data_by_index, data_by_key)

    def test_show_calls(self):
        """Test matplotlib show calls"""
        fig = self.stats_obj.show(0)
        self.stats_obj.close(fig)

        corr_mat = self.stats_obj.correlation_matrix(corr_type="Spearman")
        fig = corr_mat.show()
        corr_mat.close(fig)

        ucc = self.stats_obj.univariate_control_charts()
        fig = ucc[0].show()
        ucc[0].close(fig)

        ht2 = self.stats_obj.hotelling_t2()
        fig = ht2.show()
        ht2.close(fig)

    def test_pca(self):
        """Test PCA initialization and plot"""
        stats_obj = stats.DVHAStats(self.expected_dict_no_nan)
        pca = stats_obj.pca()
        fig = pca.show()
        pca.close(fig)

        # Test no transform
        stats_obj.pca(transform=False)

        # Test var_names=None
        stats.PCA(stats_obj.data, var_names=None)

    def test_del_const_var(self):
        """Test init deletes constant variables if del_const_vars is True"""
        stats_obj = stats.DVHAStats(
            self.const_data,
            var_names=self.expected_var_names,
            del_const_vars=True,
        )
        self.assertEqual(stats_obj.deleted_vars, ["V1"])
        assert_array_equal(
            stats_obj.data, np.delete(self.const_data, 0, axis=1)
        )

    def test_control_chart_if_const_data(self):
        """Test that const data does not crash control chart"""
        stats_obj = stats.DVHAStats(self.const_data)
        ucc = stats_obj.univariate_control_charts(box_cox=True)
        ucc[0].show()
        self.assertEqual(
            ucc[0].plot_title,
            "Cannot calculate control chart with const data!",
        )

    def test_box_cox_const_policy_raise_(self):
        """Test const_policy='raise' results in ValueError with const data"""
        stats_obj = stats.DVHAStats(self.const_data)
        with self.assertRaises(ValueError):
            stats_obj.box_cox_by_index(0, const_policy="raise")


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
