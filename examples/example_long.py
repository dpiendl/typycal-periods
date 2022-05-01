import pathlib

import matplotlib.pyplot as plt
import numpy as np

import typycal_periods as tp


# Load the measurement data
measurement_path = pathlib.Path(__file__).parent.joinpath("res/example_measurement_long.npy")
time_series_raw = np.load(str(measurement_path))
print(f"Shape of original time series: {time_series_raw.shape}")

# Plot the original time series
figure, axes = plt.subplots(2, 1, sharex="col")
axes[0].plot(time_series_raw[:, :, 0].ravel() / 1e3, color="red", linewidth=0.5)
axes[0].set_title("Energy demand")
axes[0].set_ylabel("Power in kW")
axes[1].plot(time_series_raw[:, :, 1].ravel(), color="blue", linewidth=0.5)
axes[1].set_title("Ambient temperature")
axes[1].set_xlabel("Time in h")
axes[1].set_ylabel("Temperature in °C")
for ax_i in figure.axes:
    ax_i.autoscale(enable=True, axis="x", tight=True)
    ax_i.grid()
figure.tight_layout()
plt.show()

# Store the time series in an object with the given attribute names
time_series_observations = tp.TimeSeriesObservations(time_series_raw, ("Energy demand", "Ambient temperature"))

# Transform the data to a mean of 0 with the given target standard deviation.
# This is done to weight the energy demand more than the ambient temperature .
normalizer = tp.TimeSeriesNormalizer()
normalizer.fit_normalize(time_series_observations, std_target=(2, 1))

# Cluster and evaluate up to 20 typical periods
config = tp.TypicalPeriodsSelectionConfig(pareto_error=0.2, gamma=0.1)
typical_periods_selector = tp.TypicalPeriodsSelection(time_series_observations, 20, config)
typical_periods_selector.fit()
typical_periods_selector.evaluate()

print(
    f"Minimum number of clusters: {typical_periods_selector.n_min}.\n"
    f"Optimal number of clusters: {typical_periods_selector.n_opt}."
)

# Plot the performance measure and indicator values
indicators_visualizer = tp.SelectionVisualizer(typical_periods_selector)
indicators_visualizer.plot_measures()
indicators_visualizer.plot_indicators()
indicators_visualizer.plot_indicator_improvement()

# Get the optimal number of typical periods, the labels and the periods themselves
n_opt = typical_periods_selector.n_opt
labels = typical_periods_selector.typical_periods_result(n_opt).labels
typical_periods = typical_periods_selector.typical_periods(n_opt)
# Revert the normalization
normalizer.revert_normalization(typical_periods)
normalizer.revert_normalization(time_series_observations)
# Plot the optimal number of typical periods
typical_periods_visualizer = tp.TypicalPeriodsVisualizer(time_series_observations, typical_periods, labels)
typical_periods_visualizer.plot_all()
