#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# plot.py
"""Plotting for DVHA-Stats."""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats

import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(
        self,
        y,
        show=True,
        title="Chart",
        xlabel="Independent Variable",
        ylabel="Dependent Variable",
        line=True,
        line_color=None,
        line_width=1.0,
        scatter=True,
        scatter_color=None,
    ):
        self.x = np.linspace(1, len(y), len(y))
        self.y = y
        self.show = show
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.line = line
        self.line_color = line_color
        self.line_width = line_width
        self.scatter = scatter
        self.scatter_color = scatter_color

        self.__add_labels()

        self.add_data()

        if show:
            plt.show()

    def __add_labels(self):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

    def add_data(self):
        if self.scatter:
            self.add_scatter()
        if self.line:
            self.add_default_line()

    def add_scatter(self):
        plt.scatter(self.x, self.y, color=self.scatter_color)

    def add_default_line(self):
        plt.plot(
            self.x,
            self.y,
            color=self.line_color,
            linewidth=self.line_width,
            linestyle="-",
        )

    @staticmethod
    def add_line(x, y, line_color=None, line_width=None, line_style=None):
        plt.plot(
            x,
            y,
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
        )


class ControlChart(Plot):
    def __init__(
        self,
        y,
        out_of_control,
        center_line,
        lcl=None,
        ucl=None,
        title="Control Chart",
        xlabel="Observation",
        ylabel="Charting Variable",
        line_color="black",
        line_width=0.75,
        scatter_color=None,
        **kwargs
    ):
        self.center_line = center_line
        self.lcl = lcl
        self.ucl = ucl
        self.out_of_control = out_of_control
        kwargs["title"] = title
        kwargs["xlabel"] = xlabel
        kwargs["ylabel"] = ylabel
        kwargs["line_color"] = line_color
        kwargs["line_width"] = line_width
        kwargs["scatter_color"] = scatter_color
        Plot.__init__(self, y, **kwargs)

        if "line_color" not in list(kwargs):
            self.line_color = "black"

        if self.show:
            plt.show()

    def __set_y_scatter_data(self):
        include = np.full(len(self.y), True)
        for i in self.out_of_control:
            include[i] = False
        self.ic = {"x": self.x[include], "y": self.y[include]}
        self.ooc = {"x": self.x[~include], "y": self.y[~include]}

    def add_data(self):
        super().add_data()
        self.add_control_limit_line(self.ucl)
        self.add_control_limit_line(self.lcl)
        self.add_center_line()

    def add_scatter(self):
        self.__set_y_scatter_data()
        plt.scatter(self.ic["x"], self.ic["y"], color=self.scatter_color)
        plt.scatter(self.ooc["x"], self.ooc["y"], color="red")

    def add_control_limit_line(self, limit):
        if limit is not None:
            plt.plot(
                [1, len(self.x)],
                [limit] * 2,
                color="red",
                linewidth=1.0,
                linestyle="--",
            )

    def add_center_line(self):
        plt.plot(
            [1, len(self.x)],
            [self.center_line] * 2,
            color="black",
            linewidth=1.0,
            linestyle="--",
        )
