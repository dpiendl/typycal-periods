from abc import ABCMeta
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator

from typycal_periods.clustering.selection import TypicalPeriodsSelection
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class Visualizer:
    """The base class for the visualizer.\
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self._figure: mpl.figure.Figure = plt.figure()

    def show(self):
        """Show the current matplotlib figure.\
        """
        self._figure.show()
        return

    @property
    def figure(self) -> mpl.figure.Figure:
        """The current figure, containing multiple subplots.\
        """
        return self._figure


class TypicalPeriodsVisualizer(Visualizer):
    """A visualizer for the typical periods created through the clustering.\
    """

    _mpl_settings = {
        "font.size": 8,
        "lines.linewidth": 1,
        "axes.autolimit_mode": "round_numbers",
        "axes.grid": True,
        "figure.titlesize": "x-large",
    }

    _x_label = "Time"

    def __init__(
        self,
        time_series_observations: TimeSeriesObservations,
        typical_periods: TypicalPeriods,
        labels: npt.NDArray[Any],
    ):
        """Constructor method.

        :param time_series_observations: The original time series observations.
        :param typical_periods: The typical periods created through clustering of the original time series observations.
        :param labels: The labels assigning each time series observation to a typical period. Must have a length equal
            to the number of time series observations and integers starting at 0 as values.
        """
        super().__init__()
        self._tso: TimeSeriesObservations = time_series_observations
        self._tp: TypicalPeriods = typical_periods
        self._labels: npt.NDArray[Any] = labels
        return

    def plot_all(self):
        """Creates and shows a matplotlib figure containing a subplot for every typical period with the time series
        observations it represents.
        """
        self.render_all()
        self.show()
        return

    def render_all(self) -> mpl.figure.Figure:
        """Creates and returns a matplotlib figure containing a subplot for every typical period with the time series
        observations it represents.

        :return: A matplotlib figure showing the typical periods and the assigned time series observations.
        """
        figure, axes = plt.subplots(
            self._tp.n_typical_periods,
            self._tp.n_attributes,
            sharex="col",
            figsize=self._figure_size(self._tp.n_typical_periods, self._tp.n_attributes),
        )

        if self._is_only_one_typical_period():
            self._render_single_subplot(axes, 0)
        else:
            for typical_period_index in range(self._tp.n_typical_periods):
                self._render_single_subplot(axes[typical_period_index], typical_period_index)

        self._format(figure, axes)
        self._figure = figure
        return figure

    def _is_only_one_typical_period(self):
        return self._tp.n_typical_periods == 1

    def _is_only_one_attribute(self):
        return self._tp.n_attributes == 1

    def _render_single_subplot(self, subplot, typical_period_index: int):
        mask = self._labels == typical_period_index
        for obs_j in np.where(mask)[0]:
            if self._is_only_one_attribute():
                self._render_observation(subplot, self._tso.get(obs_j))
            else:
                for attr_k in range(self._tp.n_attributes):
                    self._render_observation(subplot[attr_k], self._tso.get(obs_j, None, attr_k))

        if self._is_only_one_attribute():
            self._render_typical_period(subplot, self._tp.get(typical_period_index), label=self._tp.attribute_names[0])
        else:
            for attr_j in range(self._tp.n_attributes):
                self._render_typical_period(
                    subplot[attr_j],
                    self._tp.get(typical_period_index, None, attr_j),
                    label=self._tp.attribute_names[attr_j],
                )

        return

    def _render_observation(self, subplot, time_series: npt.NDArray[Any]):
        with mpl.rc_context(rc=self._mpl_settings):
            subplot.plot(time_series, alpha=0.5, color="k", linewidth=1)
        return

    def _render_typical_period(self, subplot, time_series: npt.NDArray[Any], label: str):
        with mpl.rc_context(rc=self._mpl_settings):
            subplot.plot(time_series, color="r", linewidth=2, label=label)
        return

    def plot_single(self, typical_period_index: int):
        """Creates and shows a matplotlib figure for the designated typical period with the time series
        observations it represents.

        :param typical_period_index: The index of the typical period to plot.
        """
        self.render_single(typical_period_index)
        self.show()
        return

    def render_single(self, typical_period: int) -> mpl.figure.Figure:
        """Creates and returns a matplotlib figure for the designated typical period with the time series
        observations it represents.

        :param typical_period: The typical period to plot.
        :return: A matplotlib figure showing the typical periods and the assigned time series observations.
        """
        self._figure, ax = plt.subplots(
            1, self._tp.n_attributes, sharex="col", figsize=self._figure_size(1, self._tp.n_attributes)
        )

        typical_period_index = typical_period - 1
        self._render_single_subplot(ax, typical_period_index)

        self._format_single_typical_period(ax, typical_period_index)
        return self._figure

    def _format(self, figure: mpl.figure.Figure, axes: Union[plt.subplot, npt.NDArray[Any]]):
        if self._is_only_one_typical_period():
            self._format_single_typical_period(axes, 0)
        else:
            for typical_period_index, ax_row in enumerate(axes):
                self._format_single_typical_period(ax_row, typical_period_index)
        figure.tight_layout()
        return

    def _format_single_typical_period(self, axes, typical_period_index: int):
        with mpl.rc_context(rc=self._mpl_settings):
            if self._is_only_one_attribute():
                self._format_single_attribute(axes, typical_period_index)
            else:
                for attr_j in range(self._tp.n_attributes):
                    self._format_single_attribute(axes[attr_j], typical_period_index)
        return

    def _format_single_attribute(self, axes, typical_period_index: int):
        axes.set_title(
            f"Typical period {typical_period_index+1}, "
            f"weight: {np.round(self._tp.weights[typical_period_index], 3)}"
        )
        axes.set_xlim(0, self._tp.len_observation - 1)
        axes.grid(visible=True, which="major", linewidth=0.5)
        axes.grid(visible=True, which="minor", axis="x", linewidth=0.2)
        axes.legend()
        if typical_period_index == self._tp.n_typical_periods - 1:
            axes.set_xlabel(self._x_label)
        return

    @staticmethod
    def _figure_size(n_rows, n_cols) -> Tuple[float, float]:
        height = 2.0 * n_rows
        width = 4.0 * n_cols
        return width, height


class SelectionVisualizer(Visualizer):
    """A visualizer for the typical periods clustering result.\
    """

    indicator_names = (
        "Profile deviation",
        "Average value deviation",
        "Load duration curve error",
        "Maximum load duration\ncurve difference",
        "Periods with an error\nabove the threshold",
    )

    _indicator_attr_names = (
        "profile_deviation",
        "average_value_deviation",
        "load_duration_curve_error",
        "maximum_load_duration_curve_difference",
        "periods_over_error_threshold",
    )

    measure_names = (
        "Average squared error",
        "Average inter-cluster distance",
        "Ratio of observed to expected squared error",
    )

    _measure_attr_names = (
        "average_squared_error",
        "average_inter_cluster_distance",
        "ratio_observed_expected_squared_error",
    )

    _mpl_settings = {
        "font.size": 8,
        "lines.linewidth": 1,
        "axes.autolimit_mode": "round_numbers",
        "axes.grid": True,
        "figure.titlesize": "x-large",
        "lines.marker": "x",
    }

    _x_label = "Number of typical periods"

    def __init__(self, typical_periods_selection: TypicalPeriodsSelection):
        """Constructor method.

        :param typical_periods_selection: An already evaluated selector for the typical periods.
        """
        super().__init__()
        self._tp_sel: TypicalPeriodsSelection = typical_periods_selection

        self._n_attr = self._tp_sel.typical_periods(1).n_attributes
        self._attr_names = self._tp_sel.typical_periods(1).attribute_names
        return

    def plot_measures(self):
        """Creates and shows a matplotlib figure of the performance measure values over the number of typical periods.\
        """
        self.render_measures()
        self.show()
        return

    def render_measures(self) -> mpl.figure.Figure:
        """Creates and returns a matplotlib figure of the performance measure values over the number of typical periods.

        :return: A matplotlib figure showing the performance measure values over the number of typical periods.
        """

        figure, axes = self._measures_subplots()
        x_values = np.arange(1, self._tp_sel.n_max + 1)
        for i, measure_attr_name_i in enumerate(self._measure_attr_names):
            measure_values = np.empty(self._tp_sel.n_max)
            for j, n_cluster in zip(range(self._tp_sel.n_max), range(1, self._tp_sel.n_max + 1)):
                measures = self._tp_sel.typical_periods_result(n_cluster).measures
                measure_values[j] = getattr(measures, measure_attr_name_i)
            with mpl.rc_context(rc=self._mpl_settings):
                axes[i].plot(x_values, measure_values)
                axes[i].set_title(self.measure_names[i])

        with mpl.rc_context(rc=self._mpl_settings):
            self._format_measures(figure, axes, "Measure values")
        self._figure = figure
        return self._figure

    def _format_measures(self, figure: mpl.figure.Figure, axes: plt.axes, title: Optional[str] = None):
        with mpl.rc_context(rc=self._mpl_settings):
            axes[-1].set_xlabel(self._x_label)
            axes[1].set_ylabel("Values")
        self._format(figure, axes, title)
        return

    def _format(self, figure: mpl.figure.Figure, axes: plt.axes, title: Optional[str] = None):
        with mpl.rc_context(rc=self._mpl_settings):
            for ax_i in figure.axes:
                ax_i.grid(visible=True, which="major", linewidth=0.5)
                ax_i.grid(visible=True, which="minor", axis="x", linewidth=0.2)
                ax_i.autoscale(enable=True, axis="x", tight=True)
                ax_i.autoscale(enable=None, axis="y", tight=False)
                ax_i.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_i.minorticks_on()
                ax_i.yaxis.set_tick_params(which="minor", bottom=False)
                ax_i.xaxis.set_minor_locator(MaxNLocator(integer=True))
            figure.suptitle(title)
            figure.tight_layout(rect=[0, 0.0, 1, 0.97])
        return

    @staticmethod
    def _measures_subplots():
        return plt.subplots(3, 1, sharex="col", figsize=(4, 6))

    def plot_indicators(self):
        """Creates and shows a matplotlib figure of the performance indicator values over the number of typical periods.

        :return: A matplotlib figure showing the performance measure values over the number of typical periods.
        """
        self.render_indicators()
        self.show()
        return

    def render_indicators(self) -> mpl.figure.Figure:
        """Creates and returns a matplotlib figure of the performance indicator values over the number of typical
        periods.

        :return: A matplotlib figure showing the performance indicator values over the number of typical periods.
        """

        figure, axes = self._indicators_subplots()
        x_values = np.arange(1, self._tp_sel.n_max + 1)
        for i, indicator_attr_name_i in enumerate(self._indicator_attr_names):
            indicator_values = self._get_indicator_values(indicator_attr_name_i)
            self._render_indicator(axes[i], x_values, indicator_values)

        self._format_indicators(figure, axes, "Indicator values")
        self._figure = figure
        return figure

    def _get_indicator_values(self, indicator_attr_name: str):
        indicator_values = np.empty((self._tp_sel.n_max, self._n_attr))
        for index, n_cluster in zip(range(self._tp_sel.n_max), range(1, self._tp_sel.n_max + 1)):
            indicators = self._tp_sel.typical_periods_result(n_cluster).indicators
            indicator_values[index] = getattr(indicators, indicator_attr_name)
        return indicator_values

    def _render_indicator(self, subplot, x_values: npt.NDArray[Any], indicator_values: npt.NDArray[Any]):
        if self._is_only_one_attribute():
            self._render_single_attribute(subplot, x_values, indicator_values)
        else:
            self._render_all_attributes(subplot, x_values, indicator_values)
        return

    def _is_only_one_attribute(self):
        return self._tp_sel.typical_periods(1).n_attributes == 1

    def _render_single_attribute(self, subplot, x_values: npt.NDArray[Any], indicator_values_1d: npt.NDArray[Any]):
        with mpl.rc_context(rc=self._mpl_settings):
            subplot.plot(x_values, indicator_values_1d)
        return

    def _render_all_attributes(self, subplot, x_values: npt.NDArray[Any], indicator_values_2d: npt.NDArray[Any]):
        for attr_j in range(self._n_attr):
            self._render_single_attribute(subplot[attr_j], x_values, indicator_values_2d[:, attr_j])
        return

    def plot_indicator_improvement(self):
        """Creates and shows a matplotlib figure of the improvement of the performance indicator values over the
        number of typical periods.
        """
        self.render_indicator_improvement()
        self.show()
        return

    def render_indicator_improvement(self) -> mpl.figure.Figure:
        """Creates and returns a matplotlib figure of the improvement of the performance indicator values over the
        number of typical periods.

        :return: A matplotlib figure showing the performance indicator improvement over the number of typical periods.
        """

        figure, axes = self._indicators_subplots()
        x_values = np.arange(1, self._tp_sel.n_max)
        for i, indicator_name_i in enumerate(self._indicator_attr_names):
            indicator_improvement = self._get_indicator_improvement(indicator_name_i)
            self._render_indicator(axes[i], x_values, indicator_improvement)

        self._format_indicators(figure, axes, "Relative indicator improvement")
        self._figure = figure
        return figure

    def _indicators_subplots(self):
        return plt.subplots(5, self._n_attr, sharex="col", figsize=(self._n_attr * 2.5 + 1, 12))

    def _get_indicator_improvement(self, indicator_attr_name: str):
        indicator_improvement = np.empty((self._tp_sel.n_max - 1, self._n_attr))
        for j, n_cluster in zip(range(self._tp_sel.n_max - 1), range(1, self._tp_sel.n_max)):
            these_indicators = self._tp_sel.typical_periods_result(n_cluster).indicators
            next_indicators = self._tp_sel.typical_periods_result(n_cluster + 1).indicators
            improvement = next_indicators.improvement_to(these_indicators)
            indicator_improvement[j] = getattr(improvement, indicator_attr_name)
        return indicator_improvement

    def _format_indicators(self, figure: mpl.figure.Figure, axes: plt.axes, title: Optional[str] = None):
        with mpl.rc_context(rc=self._mpl_settings):
            if self._is_only_one_attribute():
                axes[0].set_title(self._attr_names[0])
                axes[-1].set_xlabel(self._x_label)
                for i, ax_i in enumerate(axes):
                    ax_i.set_ylabel(self.indicator_names[i])
            else:
                for i, ax_i in enumerate(axes[0, :]):
                    ax_i.set_title(self._attr_names[i])
                for i, ax_i in enumerate(axes[:, 0]):
                    ax_i.set_ylabel(self.indicator_names[i])
                for ax_i in axes[-1, :]:
                    ax_i.set_xlabel(self._x_label)
        self._format(figure, axes, title)
        return
