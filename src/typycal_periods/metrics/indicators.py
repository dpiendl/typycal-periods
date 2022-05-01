from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from typycal_periods.metrics.base_metric import TypicalPeriodMetric
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class PerformanceIndicator(TypicalPeriodMetric, ABC):
    """Base class for the five different performance indicators used from eq. 17 to eq. 25."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculates the value of the indicator for each attribute depending on the original observations, the typical
        periods and their mapping (labels).

        :param typical_periods: The typical periods created from the original time series observations.
        :param labels: An one-dimensional array describing the mapping of the original time series observations to their
            respective typical period. Must be integers starting at 0.
        :return: array of indicator values for each attribute.
        """


class DeviationIndicator(PerformanceIndicator, ABC):
    """Base class for the deviation indicators."""

    def __init__(self, time_series_observations: TimeSeriesObservations):
        super().__init__(time_series_observations)

        self._attr_avg_per_observation: npt.NDArray[Any] = self._get_attribute_avg_per_series()
        self._attr_avg_all_observations: npt.NDArray[Any] = self._get_attribute_avg()
        return

    def _get_attribute_avg_per_series(self):
        """Based on eq. 19\
        """
        return np.sum(self.tso.observations, axis=1) / self.tso.len_observation

    def _get_attribute_avg(self):
        """Based on eq. 20\
        """
        return np.sum(self._get_attribute_avg_per_series(), axis=0) / self.tso.n_observations

    def _attribute_avg_per_cluster_center(self, typical_periods: TypicalPeriods):
        """Based on eq. 18\
        """
        return np.sum(typical_periods.observations, axis=1) / self.tso.len_observation


class ProfileDeviation(DeviationIndicator):
    """The profile deviation indicator studies the accuracy of the original and typical periods compared to their
    average. Based on eq. 17.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:

        profile_deviation = np.zeros(self.tso.n_attributes)
        attr_avg_per_cluster_center = self._attribute_avg_per_cluster_center(typical_periods)

        for i in range(typical_periods.n_typical_periods):
            for j in range(self.tso.n_observations):
                if labels[j] == i:
                    for k in range(self.tso.len_observation):
                        profile_deviation += (
                            (
                                (self.tso.observations[j, k] - self._attr_avg_per_observation[j])
                                - (typical_periods.observations[i, k] - attr_avg_per_cluster_center[i])
                            )
                            / self._attr_avg_all_observations
                        ) ** 2

        profile_deviation = np.sqrt(profile_deviation / (self.tso.n_observations * self.tso.len_observation))
        return profile_deviation


class AverageValueDeviation(DeviationIndicator):
    """The average value derivation indicator compares the average values of the original observations and the typical
    periods. Based on eq. 21.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ldc_deviation = np.zeros(self.tso.n_attributes)
        attr_avg_per_cluster_center = self._attribute_avg_per_cluster_center(typical_periods)

        for i in range(typical_periods.n_typical_periods):
            for j in range(self.tso.n_observations):
                if labels[j] == i:
                    ldc_deviation += (
                        (self._attr_avg_per_observation[j] - attr_avg_per_cluster_center[i])
                        / self._attr_avg_per_observation[j]
                    ) ** 2

        ldc_deviation = np.sqrt(ldc_deviation / self.tso.n_observations)
        return ldc_deviation


class LoadDurationCurveIndicator(PerformanceIndicator, ABC):
    """Base class for the load duration curve indicators."""

    def _ldc_o(self):
        ldc_o = self.tso.observations.reshape(
            (self.tso.n_observations * self.tso.len_observation, self.tso.n_attributes)
        )
        ldc_o = self._sort_desc(ldc_o)
        return ldc_o

    @staticmethod
    def _sort_desc(array: npt.NDArray[Any]):
        return np.sort(array[::-1], axis=0)

    def _ldc_e(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]):
        ldc_e = np.zeros((self.tso.n_observations * self.tso.len_observation, self.tso.n_attributes))

        for i in range(self.tso.n_observations):
            ldc_e[i * self.tso.len_observation : (i + 1) * self.tso.len_observation] = typical_periods.observations[
                labels[i]
            ]

        ldc_e = self._sort_desc(ldc_e)
        return ldc_e


class LoadDurationCurveError(LoadDurationCurveIndicator):
    """The load duration curve indicator extracts the relative error between the original load duration curve and the
    load duration curve of the typical periods. Based on eq. 22.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ldc_o = self._ldc_o()
        ldc_e = self._ldc_e(typical_periods, labels)

        ldc_err = np.zeros(self.tso.n_attributes)
        ldc_err[:] = np.abs(np.sum((ldc_o - ldc_e), axis=0)) / np.sum(ldc_o, axis=0)
        return ldc_err


class MaximumLoadDurationCurveDifference(LoadDurationCurveIndicator):
    """The maximum load duration curve difference indicator describes the relative difference between the maximum values
    of the load duration curve of the original observations and of the typical periods. Based on eq. 23 with changes.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ldc_o = self._ldc_o()
        ldc_e = self._ldc_e(typical_periods, labels)

        ldc_max_diff = np.zeros(self.tso.n_attributes)
        # Added an absolute operator to cover cases where the LDC is negative after rescaling.
        ldc_max_diff[:] = np.abs((np.max(ldc_o, axis=0) - np.max(ldc_e, axis=0)) / np.max(ldc_o, axis=0))
        return ldc_max_diff


class PeriodsErrorOverThreshold(PerformanceIndicator):
    """The periods error over threshold indicator determines the number of original observations whose relative average
    error to its corresponding typical period is larger than a user-defined margin. Based on eq. 24 with changes.
    """

    def __init__(self, time_series_observations: TimeSeriesObservations, gamma: float):
        """Constructor method.

        :param time_series_observations: Original time series observations on which the indicator should be applied.
        :param gamma: Relative error margin, must be within the closed interval [0, 1].
        """
        super().__init__(time_series_observations)

        self._check_gamma(gamma)
        self._gamma: float = gamma
        return

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        periods_error_over_threshold = np.zeros(self.tso.n_attributes)
        for i in range(typical_periods.n_typical_periods):
            for j in range(self.tso.n_observations):
                if labels[j] == i:
                    # Expanded the absolute operator to cover both numerator and denominator.
                    # Added normalization by the length of the observation.
                    periods_error_over_threshold += (
                        np.sum(
                            np.abs(
                                (self.tso.observations[j] - typical_periods.observations[i]) / self.tso.observations[j]
                            ),
                            axis=0,
                        )
                        / self.tso.len_observation
                        > self._gamma
                    )
        return periods_error_over_threshold

    @staticmethod
    def _check_gamma(gamma):
        if not 0 <= gamma:
            raise ValueError(f"The value of gamma must non-negative. It was {gamma}.")
        return
