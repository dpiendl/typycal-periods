from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt

from typycal_periods.metrics.base_metric import TypicalPeriodMetric
from typycal_periods.metrics.exceptions import IndicatorError
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class Measure(TypicalPeriodMetric, ABC):
    """Base class for the measures which determine the optimal number of clusters.\
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any], *args, **kwargs) -> float:
        """Calculates the value of the measure depending on the original observations, the typical
        periods and their mapping (labels).

        :param typical_periods: The typical periods created from the original time series observations.
        :param labels: An one-dimensional array describing the mapping of the original time series observations to their
            respective typical period. Must be integers starting at 0.
        :return: value of the measure.
        """

    @staticmethod
    def _sum_of_squared_dist(center, data_points):
        return (np.abs(center - data_points) ** 2).sum()


class AverageSquaredError(Measure):
    """The average squared error measure assesses the distance between the cluster centers and their assigned
    observations, thus measuring the spread within each cluster. A low value denotes a higher similarity between the
    typical period and the assigned observations. Based on eq. 10.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any], *args, **kwargs) -> float:
        avg_squared_error = 0

        for i, observation_i in enumerate(self.tso.observations):
            avg_squared_error += self._sum_of_squared_dist(typical_periods.observations[labels[i]], observation_i)

        avg_squared_error /= typical_periods.n_observations
        return avg_squared_error


class AverageInterClusterDistance(Measure):
    """The average inter-cluster distance measures the cluster's separation. A higher value constitutes a larger
    difference between different typical periods, thus making the typical periods less similar to each other. Based on
    eq. 11.
    """

    def value(self, typical_periods: TypicalPeriods, labels: npt.NDArray[Any], *args, **kwargs) -> float:
        avg_inter_cluster_distance = 0.0

        for typical_period_i in typical_periods.observations:
            avg_inter_cluster_distance += self._sum_of_squared_dist(typical_period_i, typical_periods.observations)

        avg_inter_cluster_distance /= typical_periods.n_observations**2
        return avg_inter_cluster_distance


class RatioObservedExpectedSquaredError:
    """This measures determines the ratio of the observed to the expected error for the selected number of clusters.
    A low value means that the clustering with the selected number of clusters is better defined than with fewer
    clusters. Based on eq. 12 and 13.
    """

    def __init__(self, time_series_observations: TimeSeriesObservations):
        """Constructor method.

        :param time_series_observations: Original time series observations on which the indicator should be applied.
        """
        self._tso: TimeSeriesObservations = time_series_observations
        self._alpha: Optional[float] = None
        return

    def value(
        self,
        typical_periods: TypicalPeriods,
        avg_squared_error: float,
        avg_squared_error_n_minus_1: Optional[float] = None,
        alpha_n_minus_1: Optional[float] = None,
    ) -> float:
        """Calculates the value of the ratio of the observed to the expected squared error depending on the squared
        error for the current number of clusters, the squared error for one fewer number of clusters and the alpha
        periods and their mapping (labels).

        :param typical_periods: The typical periods created from the original time series observations.
        :param avg_squared_error: The average squared error for this number of cluster.
        :param avg_squared_error_n_minus_1: The average squared error for one fewer number of clusters. Must only be
            supplied if the current number of clusters is larger than two. Defaults to None.
        :param alpha_n_minus_1: The alpha value for one fewer number of clusters. Must only be supplied if the current
            number of clusters is larger than two. Defaults to None.
        :return: value of the ration of the observed to the expected squared error.
        """
        self._value_argument_checker(typical_periods, avg_squared_error_n_minus_1)

        if self._more_than_one_cluster(typical_periods) and avg_squared_error_n_minus_1 > 0:
            self._calc_alpha(typical_periods, alpha_n_minus_1)
            value = (
                typical_periods.n_observations
                * avg_squared_error
                / (self.alpha * (typical_periods.n_observations - 1) * avg_squared_error_n_minus_1)
            )
        elif self._more_than_one_cluster(typical_periods):
            self._alpha = 1.0
            value = 1.0
        else:
            value = 1.0

        return value

    def _value_argument_checker(
        self, typical_periods: TypicalPeriods, avg_squared_error_n_minus_1: Optional[float] = None
    ):
        if self._more_than_one_cluster(typical_periods):
            if avg_squared_error_n_minus_1 is None:
                raise IndicatorError(
                    f"When calculating the {self.__class__.__name__} value for more than one "
                    f"cluster, you must supply the average squared error result with one less "
                    f"cluster, but avg_squared_error_n_minus_1 was {avg_squared_error_n_minus_1}."
                )
        return

    @staticmethod
    def _more_than_one_cluster(typical_periods: TypicalPeriods):
        return typical_periods.n_observations > 1

    def _calc_alpha(self, typical_periods: TypicalPeriods, alpha_n_minus_1: float):
        self._alpha_argument_check(typical_periods, alpha_n_minus_1)

        if self._two_clusters(typical_periods):
            self._alpha = 1 - 3 / (4 * self.tso.n_attributes * self.tso.len_observation)
        elif self._more_than_two_clusters(typical_periods):
            self._alpha = alpha_n_minus_1 + (1 - alpha_n_minus_1) / 6
        return

    @property
    def alpha(self):
        """Get the alpha value of the ratio of the observed to the expected squared error.\
        """
        return self._alpha

    def _alpha_argument_check(self, typical_periods: TypicalPeriods, alpha_n_minus_1: float):
        if self._single_cluster(typical_periods):
            raise IndicatorError("The alpha value cannot be calculated with only a single cluster.")
        if not self._more_than_one_observed_value():
            raise IndicatorError(
                f"When calculating the alpha value, the number of attributes "
                f"({self.tso.n_attributes}) multiplied by the observation length "
                f"({self.tso.len_observation}) must be greater than 1."
            )
        if self._more_than_two_clusters(typical_periods) and (alpha_n_minus_1 is None):
            raise IndicatorError(
                f"For calculating the alpha value with more than two clusters, the alpha value for "
                f"the result with one less cluster must be supplied, but was {alpha_n_minus_1}."
            )
        return

    @staticmethod
    def _two_clusters(typical_periods: TypicalPeriods):
        return typical_periods.n_typical_periods == 2

    @staticmethod
    def _more_than_two_clusters(typical_periods: TypicalPeriods):
        return typical_periods.n_typical_periods > 2

    @staticmethod
    def _single_cluster(typical_periods: TypicalPeriods):
        return typical_periods.n_typical_periods == 1

    def _more_than_one_observed_value(self):
        return self.tso.n_observations * self.tso.n_attributes > 1

    @property
    def tso(self) -> TimeSeriesObservations:
        """Get the time series observations.\
        """
        return self._tso
