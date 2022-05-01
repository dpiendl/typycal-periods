from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

import numpy.typing as npt

from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class TypicalPeriodMetric(ABC):
    """Base class for the measures and performance indicators used to determine the minimum and optimal number of
    clusters.
    """

    __metaclass__ = ABCMeta

    def __init__(self, time_series_observations: TimeSeriesObservations):
        """ "Constructor method.

        :param time_series_observations: Original time series observations on which the indicator should be applied.
        """
        self._tso: TimeSeriesObservations = time_series_observations
        return

    @abstractmethod
    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any], *args, **kwargs) -> Any:
        """Calculates the value of the indicator for each attribute depending on the original observations, the typical
        periods and their mapping (labels).

        :param typical_periods: The typical periods created from the original time series observations.
        :param labels: An one-dimensional array describing the mapping of the original time series observations to their
            respective typical period. Must be integers starting at 0.
        :return: value of the metric.
        """

    @property
    def tso(self) -> TimeSeriesObservations:
        """Get the time series observations.\
        """
        return self._tso
