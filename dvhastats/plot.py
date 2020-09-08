#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# plot.py
"""Basic plotting class objects for DVHA-Stats based on matplotlib"""
#
# Copyright (c) 2020 Dan Cutright
# This file is part of DVHA-Stats, released under a MIT license.
#    See the file LICENSE included with this distribution, also
#    available at https://github.com/cutright/DVHA-Stats

from matplotlib import pyplot as plt
import numpy as np


FIGURE_COUNT = 1


def get_new_figure_num():
    """Get a number for a new matplotlib figure

    Returns
    ----------
    int
        Figure number
    """
    global FIGURE_COUNT
    FIGURE_COUNT += 1
    return FIGURE_COUNT - 1


class Chart:
    """Base class for charts"""

    def __init__(self, title=None, fig_init=True):
        """Initialization of Chart base class"""
        self.title = title
        self.figure = plt.figure(get_new_figure_num()) if fig_init else None

        if title and fig_init:
            self.figure.suptitle(title, fontsize=16)

    def show(self):
        """Display this figure"""
        self.activate()
        plt.show()

    def activate(self):
        """Activate this figure"""
        plt.figure(self.figure.number)

    def close(self):
        """Close this figure"""
        plt.close(self.figure.number)


class Plot(Chart):
    """Generic plotting class with matplotlib"""

    def __init__(
        self,
        y,
        x=None,
        show=True,
        title="Chart",
        xlabel="Independent Variable",
        ylabel="Dependent Variable",
        line=True,
        line_color=None,
        line_width=1.0,
        line_style="-",
        scatter=True,
        scatter_color=None,
    ):
        """
        Initialization of a general Plot class object

        Parameters
        ----------
        y : np.ndarray, list
            The y data to be plotted (1-D only)
        x : np.ndarray, list, optional
            Optionally specify the x-axis values. Otherwise index+1 is used.
        show : bool
            Automatically plot the data if True
        title : str
            Set the plot title
        xlabel : str
            Set the x-axis title
        ylabel : str
            Set the y-axis title
        line : bool
            Plot the data as a line series
        line_color : str, optional
            Specify the line color
        line_width : float, int
            Specify the line width
        line_style : str
            Specify the line style
        scatter : bool
            Plot the data as a scatter plot (circles)
        scatter_color : str, optional
            Specify the scatter plot circle color
        """
        Chart.__init__(self, title=title)
        self.x = np.linspace(1, len(y), len(y)) if x is None else np.array(x)
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.show = show
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.line = line
        self.line_color = line_color
        self.line_width = line_width
        self.line_style = line_style
        self.scatter = scatter
        self.scatter_color = scatter_color
        self.activate()

        self.__add_labels()
        self.__add_data()

        if show:
            plt.show()

    def __add_labels(self):
        """Set the x and y axes labels to figure"""
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def __add_data(self):
        """Add scatter and/or line data to figure"""
        if self.scatter:
            self.add_scatter()
        if self.line:
            self.add_default_line()

    def add_scatter(self):
        """Add scatter data to figure"""
        self.activate()
        plt.scatter(self.x, self.y, color=self.scatter_color)

    def add_default_line(self):
        """Add line data to figure"""
        self.activate()
        plt.plot(
            self.x,
            self.y,
            color=self.line_color,
            linewidth=self.line_width,
            linestyle=self.line_style,
        )

    def add_line(
        self, y, x=None, line_color=None, line_width=None, line_style=None
    ):
        """Add another line with the provided data

        Parameters
        ----------
        y : np.ndarray, list
            The y data to be plotted (1-D only)
        x: np.ndarray, list, optional
            Optionally specify the x-axis values. Otherwise index+1 is used.
        line_color: str, optional
            Specify the line color
        line_width: float, int
            Specify the line width
        line_style : str
            Specify the line style
        """
        self.activate()
        plt.plot(
            np.linspace(1, len(y), len(y)) if x is None else x,
            y,
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
        )


class ControlChart(Plot):
    """ControlChart class object"""

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
        center_line_color="black",
        center_line_width=1.0,
        center_line_style="--",
        limit_line_color="red",
        limit_line_width=1.0,
        limit_line_style="--",
        **kwargs
    ):
        """Initialization of a ControlChart plot class object

        Parameters
        ----------
        y : np.ndarray, list
            Charting data
        out_of_control : np.ndarray, list
            The indices of y that are out-of-control
        center_line : float, np.ndarray
            The center line value (e.g., np.mean(y))
        lcl : float, optional
            The lower control limit (LCL). Line omitted if lcl is None.
        ucl : float, optional
            The upper control limit (UCL). Line omitted if ucl is None.
        title: str
            Set the plot title
        xlabel: str
            Set the x-axis title
        ylabel: str
            Set the y-axis title
        line_color: str, optional
            Specify the line color
        line_width: float, int
            Specify the line width
        kwargs : any
            Any additional keyword arguments applicable to the Plot class
        """
        self.center_line = center_line
        self.lcl = lcl
        self.ucl = ucl
        self.out_of_control = out_of_control
        self.center_line_color = center_line_color
        self.center_line_width = center_line_width
        self.center_line_style = center_line_style
        self.limit_line_color = limit_line_color
        self.limit_line_width = limit_line_width
        self.limit_line_style = limit_line_style
        kwargs["title"] = title
        kwargs["xlabel"] = xlabel
        kwargs["ylabel"] = ylabel
        kwargs["line_color"] = line_color
        kwargs["line_width"] = line_width
        Plot.__init__(self, y, **kwargs)
        self.__add_cc_data()
        self.__add_table_with_limits()

        if self.show:
            plt.show()

    def __set_y_scatter_data(self):
        """Add circles colored by out-of-control status"""
        include = np.full(len(self.y), True)
        for i in self.out_of_control:
            include[i] = False
        self.ic = {"x": self.x[include], "y": self.y[include]}
        self.ooc = {"x": self.x[~include], "y": self.y[~include]}

    def __add_cc_data(self):
        """Add center line and upper/lower control limit lines"""
        self.add_control_limit_line(self.ucl)
        self.add_control_limit_line(self.lcl)
        self.add_center_line()

    def __add_table_with_limits(self):
        """Add tables with center line and upper/lower control limit values"""
        self.activate()
        plt.subplots_adjust(bottom=0.25)
        plt.table(
            cellText=self.__table_text,
            cellLoc="center",
            colLabels=["Center Line", "LCL", "UCL"],
            loc="bottom",
            bbox=[0.0, -0.31, 1, 0.12],
        )

    @property
    def __table_text(self):
        """Get text to pass into matplotlib table creation"""
        props = ["center_line", "lcl", "ucl"]
        text = []
        for prop in props:
            value = getattr(self, prop)
            if isinstance(value, float):
                formatter = ["E", "f"][9999 > float(value) > 0.1]
                text.append(("%%0.3%s" % formatter) % value)
            else:
                text.append(str(value))
        return [text]

    def add_scatter(self):
        """Set scatter data, add in- and out-of-control circles"""
        self.activate()
        self.__set_y_scatter_data()
        plt.scatter(self.ic["x"], self.ic["y"], color=self.scatter_color)
        plt.scatter(self.ooc["x"], self.ooc["y"], color="red")

    def add_control_limit_line(
        self, limit, color=None, line_width=None, line_style=None
    ):
        """Add a control limit line to plot"""
        self.activate()
        color = self.limit_line_color if color is None else color
        line_width = (
            self.limit_line_width if line_width is None else line_width
        )
        line_style = (
            self.limit_line_style if line_style is None else line_style
        )
        if limit is not None:
            plt.plot(
                [1, len(self.x)],
                [limit] * 2,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
            )

    def add_center_line(self, color=None, line_width=None, line_style=None):
        """Add the center line to the plot"""
        self.activate()
        color = self.center_line_color if color is None else color
        line_width = (
            self.center_line_width if line_width is None else line_width
        )
        line_style = (
            self.center_line_style if line_style is None else line_style
        )
        plt.plot(
            [1, len(self.x)],
            [self.center_line] * 2,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
        )


class HeatMap(Chart):
    """Create a heat map using matplotlib.pyplot.matshow"""

    def __init__(
        self,
        X,
        xlabels=None,
        ylabels=None,
        title=None,
        cmap="viridis",
        show=True,
    ):
        """Initialization of a HeatMap Chart object"""
        Chart.__init__(self, title=title)
        self.X = X
        self.x_labels = range(X.shape[1]) if xlabels is None else xlabels
        self.y_labels = range(X.shape[0]) if ylabels is None else ylabels

        plt.matshow(X, cmap=cmap, fignum=self.figure.number)
        plt.colorbar()
        self.__set_ticks()

        if show:
            self.show()

    def __set_ticks(self):
        """Set tick labels based on x and y labels"""
        plt.xticks(
            range(self.X.shape[1]), self.x_labels, rotation=30, ha="left"
        )
        plt.yticks(range(self.X.shape[0]), self.y_labels, rotation=30)


class PCAFeatureMap(HeatMap):
    """Specialized Heat Map for PCA feature evaluation"""

    def __init__(
        self,
        X,
        features=None,
        cmap="viridis",
        show=True,
        title="PCA Feature Heat Map",
    ):
        """Initialization of a HeatMap Chart object"""
        HeatMap.__init__(
            self,
            X,
            xlabels=features,
            ylabels=self.get_comp_labels(X.shape[0]),
            cmap=cmap,
            show=show,
            title=title,
        )

    def get_comp_labels(self, n_components):
        """Get ylabels for HeatMap"""
        return [
            "%s Comp" % (self.get_ordinal(n + 1)) for n in range(n_components)
        ]

    @staticmethod
    def get_ordinal(n):
        """Convert number to its ordinal (e.g., 1 to 1st)

        Parameters
        ----------
        n : int
            Number to be converted to ordinal

        Returns
        ----------
        str
            the ordinal of n
        """
        return "%d%s" % (
            n,
            "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )


class Histogram(Chart):
    """Histogram plotting class object"""

    def __init__(
        self,
        data,
        bins=10,
        title="Histogram",
        xlabel="Bins",
        ylabel="Counts",
        **kwargs
    ):
        """Initialization of Histogram class

        Parameters
        ----------
        data : array-like
            Input array (1-D)
        bins : int, sequence, str
            default: rcParams["hist.bins"] (default: 10)
            If bins is an integer, it defines the number of equal-width
            bins in the range.

            If bins is a sequence, it defines the bin edges, including the
            left edge of the first bin and the right edge of the last bin;
            in this case, bins may be unequally spaced. All but the last
            (righthand-most) bin is half-open. In other words, if bins is:
            [1, 2, 3, 4]
            then the first bin is [1, 2) (including 1, but excluding 2) and
            the second [2, 3). The last bin, however, is [3, 4], which
            includes 4.

            If bins is a string, it is one of the binning strategies supported
            by numpy.histogram_bin_edges: 'auto', 'fd', 'doane', 'scott',
            'stone', 'rice', 'sturges', or 'sqrt'.
        kwargs : any
            Any keyword argument may be set per matplotlib histogram:
            https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.hist.html

        """
        self.title = title
        Chart.__init__(self, title=self.title, fig_init=False)

        self.data = data
        self.bins = bins
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.hist_kwargs = kwargs

        self.__set_hist_data()
        self.__set_title()
        self.__add_labels()

    def __set_hist_data(self):
        """Generate histogram data and add to figure"""
        self.figure, self.axes = plt.subplots()
        self.axes.hist(self.data, bins=self.bins, **self.hist_kwargs)

    def __set_title(self):
        """Set the figure title"""
        self.figure.suptitle(self.title, fontsize=16)

    def __add_labels(self):
        """Set the x and y axes labels to figure"""
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
