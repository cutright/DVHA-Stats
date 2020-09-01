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
from numpy import random
import matplotlib

matplotlib.use("Template")


class TestPlot(unittest.TestCase):
    """Unit tests for Utilities."""

    def setUp(self):
        """Setup files and base data for utility testing."""
        # self.x = [0, 1]
        self.y = random.standard_normal(10)

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
            random.randint(5, size=3),
            0,
            lcl=-1,
            ucl=1,
        )
        p.close()


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
