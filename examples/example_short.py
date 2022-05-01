import pathlib

import numpy as np

import typycal_periods as tp


# Load the time series data of the shape (50, 51, 1)
measurement_path = pathlib.Path(__file__).parent.joinpath("res/example_measurement_short.npy")
time_series_raw = np.load(str(measurement_path))

# Put it in an TimeSeriesObservations object
observations = tp.TimeSeriesObservations(time_series_raw)

# Start the clustering and evaluation process
selector = tp.TypicalPeriodsSelection(observations, 10)
selector.fit()
selector.evaluate()

# Plot the performance measures and indicators
indicator_visualizer = tp.SelectionVisualizer(selector)
indicator_visualizer.plot_measures()
indicator_visualizer.plot_indicators()

# Select the desired number of typical periods based on the visual inspection
n_user = 4
typical_periods_visualizer = tp.TypicalPeriodsVisualizer(
    observations, selector.typical_periods(n_user), selector.typical_periods_result(n_user).labels
)
typical_periods_visualizer.plot_all()

# Alternatively, let the selector choose the optimal number of typical periods based on the performance measures and
# indicator values.
n_opt = selector.n_opt
typical_periods_visualizer = tp.TypicalPeriodsVisualizer(
    observations, selector.typical_periods(n_opt), selector.typical_periods_result(n_opt).labels
)
typical_periods_visualizer.plot_all()
