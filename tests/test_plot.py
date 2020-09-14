#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# test_plot.py
"""unittest cases for plot."""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats


import unittest
from dvhastats import plot
import numpy as np
import matplotlib

matplotlib.use("Template")


class TestPlot(unittest.TestCase):
    """Unit tests for Utilities."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        # self.x = [0, 1]
        self.y = np.random.standard_normal(10)

    def test_plot_init(self):
        """Test Plot initialization"""
        p = plot.Plot(self.y)
        p.close()

    def test_plot_add_line(self):
        """Test add_line and show=False"""
        p = plot.Plot(self.y, show=False)
        p.add_line([0, 1])
        p.close()

    def test_control_chart(self):
        """Test ControlChart initialization"""
        p = plot.ControlChart(
            self.y,
            np.random.randint(5, size=3),
            0,
            lcl=-1,
            ucl=1,
        )
        p.close()

    def test_histogram(self):
        """Test Histogram initialization"""
        p = plot.Histogram(self.y)
        p.close()

    def test_boxplot(self):
        """Test Histogram initialization"""
        # 1D
        p = plot.BoxPlot(self.y, xlabels=[1])
        p.close()
        # 2D
        p = plot.BoxPlot(np.array([self.y, self.y]).T, xlabels=[1, 2])
        p.close()


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
