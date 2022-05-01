import numpy as np
import pytest

import typycal_periods as tp


def build_observations(n_attrs):
    np.random.seed(0)
    length = 50
    n_series = 50
    t = np.linspace(0, 2 * np.pi, length)
    raw_data = np.zeros((n_series, length, n_attrs))
    for j in range(n_attrs):
        shift = np.random.random(n_series) * length / 2
        for i in range(n_series):
            raw_data[i, :, j] = np.sin(t + shift[i])
    observations = tp.TimeSeriesObservations(raw_data)
    return observations


@pytest.fixture(params=((1, 5), (2, 5), (1, 1), (2, 1)), scope="session")
def typical_days_result(request):
    n_attrs, n_typical_periods = request.param
    observations = build_observations(n_attrs)
    selector = tp.TypicalPeriodsSelection(observations, n_typical_periods)
    selector.fit()
    selector.evaluate()
    return observations, selector


@pytest.mark.slow
def test_visualize_tp(typical_days_result):
    observations, selector = typical_days_result
    tp_visualizer = tp.TypicalPeriodsVisualizer(
        observations, selector.typical_periods(selector.n_max), selector.typical_periods_result(selector.n_max).labels
    )
    tp_visualizer.plot_all()


@pytest.mark.slow
def test_visualize_tp_single(typical_days_result):
    observations, selector = typical_days_result
    tp_visualizer = tp.TypicalPeriodsVisualizer(
        observations, selector.typical_periods(selector.n_max), selector.typical_periods_result(selector.n_max).labels
    )
    tp_visualizer.render_single(selector.n_max)
    _ = tp_visualizer.figure
    tp_visualizer.plot_single(selector.n_max)


@pytest.mark.slow
def test_visualize_indicators(typical_days_result):
    selector = typical_days_result[1]
    indicator_visualizer = tp.SelectionVisualizer(selector)
    indicator_visualizer.plot_measures()
    indicator_visualizer.plot_indicators()
    indicator_visualizer.plot_indicator_improvement()
