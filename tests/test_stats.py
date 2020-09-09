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
from dvhastats.utilities import import_data
import warnings
from copy import deepcopy


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
        self.expected_arr = np.array(data).T
        self.expected_arr_no_nan = deepcopy(self.expected_arr)
        self.expected_arr_no_nan[0, 0] = 56.5
        self.expected_var_names = keys

        self.const_data = deepcopy(self.expected_arr)
        self.const_data[:, 0] = np.ones_like(self.expected_arr[:, 0])

        self.data, self.var_names = import_data(self.expected_dict)

    def test_arr_import(self):
        """Test array importing"""
        data, var_names = import_data(
            self.expected_arr, self.expected_var_names
        )
        assert_array_equal(data, self.expected_arr)
        self.assertEqual(var_names, self.expected_var_names)

    def test_dict_import(self):
        """Test dict importing"""
        data, var_names = import_data(self.expected_dict)
        assert_array_equal(data, self.expected_arr)
        self.assertEqual(var_names, self.expected_var_names)

    def test_csv_import(self):
        """Test csv importing"""
        data, var_names = import_data(self.data_path)
        assert_array_equal(data, self.expected_arr)
        self.assertEqual(var_names, self.expected_var_names)

    def test_invalid_data_import(self):
        with self.assertRaises(NotImplementedError):
            import_data("test")

    def test_correlation_matrix_type_failure(self):
        """Check that error is raised if corr_type Spearman or Pearson"""
        with self.assertRaises(NotImplementedError):
            stats.CorrelationMatrix(self.data, corr_type="test")

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

        corr_mat = stats.CorrelationMatrix(self.data)
        warnings.filterwarnings("ignore")
        data = corr_mat.chart_data
        warnings.filterwarnings("default")
        assert_array_almost_equal(data["corr"], exp_r)
        assert_array_almost_equal(data["p"], exp_p)

        exp_norm = np.array(
            [
                5.560821491105694,
                2.168424114591037,
                0.4849069523738952,
                0.36635238844506546,
                3.997163488853074,
                5.497952110420669,
            ]
        )
        exp_norm_p = np.array(
            [
                0.06201303056894888,
                0.3381681380630833,
                0.7847002539774257,
                0.8326214397249772,
                0.1355273594307197,
                0.06399335333154248,
            ]
        )
        assert_array_almost_equal(data["norm"], exp_norm)
        assert_array_almost_equal(data["norm_p"], exp_norm_p)

        self.validate_dict_is_json_compat(data)

    def test_spearman_correlation_matrix(self):
        """Test Spearman correlation matrix calculation"""
        corr_mat = stats.CorrelationMatrix(self.data, corr_type="Spearman")
        self.assertEqual(corr_mat.corr_type, "spearman")

        warnings.filterwarnings("ignore")
        data = corr_mat.chart_data
        warnings.filterwarnings("default")
        self.validate_dict_is_json_compat(data)

    def test_control_chart(self):
        """Test univariate control chart creation and values"""
        ucc = stats.ControlChart(self.data[:, 0])
        str_rep = (
            "center_line: 48.544\ncontrol_limits: "
            "28.199, 68.890\nout_of_control: []"
        )
        self.assertEqual(str(ucc), str_rep)
        self.assertEqual(repr(ucc), str_rep)

        chart_data = ucc.chart_data
        self.assertEqual(len(chart_data["out_of_control"]), 0)
        self.assertEqual(round(chart_data["center_line"], 3), 48.544)
        self.assertEqual(round(chart_data["lcl"], 3), 28.199)
        self.assertEqual(round(chart_data["ucl"], 3), 68.89)
        self.assertEqual(len(ucc.out_of_control_high), 0)
        self.assertEqual(len(ucc.out_of_control_low), 0)

        self.validate_dict_is_json_compat(ucc.chart_data)

    def test_control_chart_with_limits(self):
        """Test univariate control chart creation and values"""
        ucc = stats.ControlChart(
            self.data[:, 0],
            lcl_limit=30,
            ucl_limit=50,
        )
        lcl, ucl = ucc.control_limits
        self.assertEqual(round(lcl, 3), 30)
        self.assertEqual(round(ucl, 3), 50)
        self.assertEqual(len(ucc.out_of_control), 1)

    def test_control_chart_with_all_nan_input(self):
        data = np.array([[np.nan] * 5]).T
        ucc = stats.ControlChart(data)
        self.assertTrue(np.isnan(ucc.center_line))

    def test_hotelling_t2(self):
        """Test multivariate control chart creation and values"""
        ht2 = stats.HotellingT2(self.data, const_policy="omit")

        chart_data = ht2.chart_data
        lcl, ucl = ht2.control_limits
        self.assertEqual(round(chart_data["center_line"], 3), 5.614)
        self.assertEqual(round(lcl, 3), 0)
        self.assertEqual(round(ucl, 3), 7.834)
        self.assertEqual(len(chart_data["out_of_control"]), 0)
        str_rep = (
            "Q: [nan nan nan nan nan nan nan nan nan nan]\n"
            "center_line: 5.614\n"
            "control_limits: 0, 7.834\n"
            "out_of_control: []"
        )
        self.assertEqual(str(ht2), str_rep)
        self.assertEqual(repr(ht2), str_rep)

        self.validate_dict_is_json_compat(ht2.chart_data)

    def test_pca(self):
        """Test PCA"""
        pca = stats.PCA(self.expected_arr_no_nan)
        exp = np.array(
            [
                [
                    0.26641603,
                    0.24490217,
                    0.32525642,
                    0.49217851,
                    0.55671381,
                    0.45943851,
                ],
                [
                    -0.55407909,
                    -0.56054476,
                    -0.38688698,
                    0.19060815,
                    0.33893785,
                    0.2790951,
                ],
            ]
        )
        assert_array_almost_equal(pca.feature_map_data, exp)
        self.assertEqual(pca.component_labels, ["1st Comp", "2nd Comp"])

        pca = stats.PCA(self.expected_arr_no_nan, transform=False)
        assert_array_almost_equal(pca.feature_map_data, exp)

    def test_multi_variable_regression(self):
        """Test Multi-Variable Linear Regression"""
        y = np.linspace(1, 10, 10)
        mvr = stats.MultiVariableRegression(self.expected_arr_no_nan, y)
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

        mvr2 = stats.MultiVariableRegression(
            self.expected_arr_no_nan, y, saved_reg=mvr
        )
        assert_array_equal(mvr.residuals, mvr2.residuals)

        self.validate_dict_is_json_compat(mvr.chart_data)
        self.validate_dict_is_json_compat(mvr.prob_plot)

        self.assertIsInstance(repr(mvr), str)

    def test_box_cox(self):
        """Test Box-Cox"""
        stats.box_cox(self.expected_arr_no_nan[:, 0])
        self.assertTrue(stats.is_nan_arr(stats.box_cox(np.ones(10))))
        with self.assertRaises(ValueError):
            stats.is_nan_arr(stats.box_cox(np.ones(10), const_policy="raise"))

    def test_moving_avg(self):
        """Test moving average calculation"""
        avg_len = 5
        y = np.array([4, 23, 1, 4, -18, 2.2, 100, 34, 9, 8])
        x = np.linspace(1, len(y), len(y))
        x_avg, y_avg = stats.moving_avg(y, avg_len)
        assert_array_equal(x_avg, x[avg_len - 1 : :])
        y_avg_exp = np.array([2.8, 2.44, 17.84, 24.44, 25.44, 30.64])
        assert_array_almost_equal(y_avg, y_avg_exp)

        avg_len = 3
        x += 10
        w = np.array([1, 2, 1, 1, 5, 2, 1, 1, 1, 1])
        x_avg, y_avg = stats.moving_avg(y, avg_len, x=x, weight=w)
        assert_array_equal(x_avg, x[avg_len - 1 : :])
        y_avg_exp = np.array(
            [5.5, 5.5, 0.466667, 0.5, 32.5, 45.033333, 47.666667, 17.0]
        )
        assert_array_almost_equal(y_avg, y_avg_exp)

    def test_avg_moving_range(self):
        """Test avg moving range"""
        test = np.array([1, 2, 3, 4, np.nan])
        self.assertEqual(stats.avg_moving_range(test), 1.0)
        test_nan = np.array([np.nan])
        self.assertTrue(np.isnan(stats.avg_moving_range(test_nan)))

    def test_is_arr_const(self):
        """Test if array is constant"""
        self.assertTrue(stats.is_arr_constant(np.ones(5)))
        self.assertFalse(stats.is_arr_constant(np.array([1, 2])))

    def test_process_nan_policy(self):
        """Test process nan policy"""
        arr = np.array([1, 2, 3, np.nan])
        test = stats.process_nan_policy(arr, "omit")
        assert_array_equal(test, arr[:3])

        with self.assertRaises(NotImplementedError):
            stats.process_nan_policy(arr, "raise")

        test = stats.process_nan_policy(arr, "propagate")
        self.assertTrue(np.isnan(test))

    def test_histogram(self):
        """Test histogram class object"""
        hist = stats.Histogram(np.linspace(1, 20, 20), 4)
        data = hist.chart_data
        assert_array_equal(data["x"], [3.375, 8.125, 12.875, 17.625])
        assert_array_equal(data["y"], [5, 5, 5, 5])
        self.assertEqual(data["mean"], 10.5)
        self.assertEqual(data["median"], 10.5)
        self.assertEqual(round(data["std"], 3), 5.766)
        self.assertEqual(round(data["normality"], 3), 3.992)
        self.assertEqual(round(data["normality_p"], 3), 0.136)

        self.validate_dict_is_json_compat(hist.chart_data)

    def validate_dict_is_json_compat(self, data_dict):
        """Verify dictionary is json compatible"""
        for key, value in data_dict.items():
            self.assertIsInstance(key, (int, str))
            if isinstance(value, list):
                self.validate_list_is_json_compat(value)
            elif isinstance(value, dict):
                self.validate_dict_is_json_compat(value)
            else:
                self.assertIsInstance(value, (float, str, int))

    def validate_list_is_json_compat(self, data_list):
        """Verify list is json compatible"""
        for item in data_list:
            if isinstance(item, list):
                self.validate_list_is_json_compat(item)
            elif isinstance(item, dict):
                self.validate_dict_is_json_compat(item)
            else:
                self.assertIsInstance(item, (float, str, int))


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
