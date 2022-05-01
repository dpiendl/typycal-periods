from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
from tslearn.clustering import TimeSeriesKMeans

from typycal_periods.metrics.exceptions import EvaluationError
from typycal_periods.metrics.exceptions import NotClusteredError
from typycal_periods.metrics.exceptions import NotEvaluatedError
from typycal_periods.metrics.indicators import AverageValueDeviation
from typycal_periods.metrics.indicators import LoadDurationCurveError
from typycal_periods.metrics.indicators import MaximumLoadDurationCurveDifference
from typycal_periods.metrics.indicators import PeriodsErrorOverThreshold
from typycal_periods.metrics.indicators import ProfileDeviation
from typycal_periods.metrics.measures import AverageInterClusterDistance
from typycal_periods.metrics.measures import AverageSquaredError
from typycal_periods.metrics.measures import RatioObservedExpectedSquaredError
from typycal_periods.tp_types.data_classes import PerformanceIndicators
from typycal_periods.tp_types.data_classes import PerformanceMeasures
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class TypicalPeriodsClustering:
    """This class clusters a time series observations using the K-Means algorithm with a euclidean metric
    and evaluates the clustering result.
    """

    def __init__(self, time_series_observations: TimeSeriesObservations, n_cluster: int, gamma: float):
        """Constructor method.

        :param time_series_observations: A TimeSeriesObservations object that should be clustered.
        :param n_cluster: Positive integer determining the number of clusters used for clustering.
        :param gamma: Float between 0 and 1 for calculating the performance indicator 'Number of periods with relative
            error higher than gamma'.
        """
        self._tso = time_series_observations
        self._n_cluster: int = int(n_cluster)
        self._gamma: float = gamma

        self._typical_periods: Optional[TypicalPeriods] = None
        self._labels: Optional[npt.NDArray[Any]] = None

        self._measures: Optional[PerformanceMeasures] = None
        self._indicators: Optional[PerformanceIndicators] = None

        self._is_clustered: bool = False
        self._is_evaluated: bool = False

        self._alpha: Optional[float] = None

        return

    def fit(self):
        """Creates typical periods from the time series observations using K-Means clustering with a euclidean metric.\
        """
        np.random.seed(0)
        estimator = TimeSeriesKMeans(n_clusters=self._n_cluster, metric="euclidean")

        self._labels = estimator.fit_predict(X=self._tso.observations)
        self._typical_periods = TypicalPeriods(
            estimator.cluster_centers_, self._tso.attribute_names, self._weights(self._labels)
        )

        self._is_clustered = True
        return

    def _weights(self, labels: npt.NDArray[Any]) -> npt.NDArray[Any]:
        weights = np.zeros(self._n_cluster)
        for i in range(self._n_cluster):
            weights[i] = np.sum(labels == i)
        weights /= self._tso.n_observations
        return weights

    def evaluate(self, clustering_n_minus_1: Optional["TypicalPeriodsClustering"] = None):
        """Evaluates the clustering result by calculating the measures and performance indicators.\
        """
        self._check_is_clustered("The time series have to be clustered before being evaluated.")
        self._check_evaluate_arguments(clustering_n_minus_1)

        self._evaluate_measures(clustering_n_minus_1)
        self._evaluate_indicators()

        self._is_evaluated = True
        return

    def _check_is_clustered(self, msg: str):
        if not self._is_clustered:
            raise NotClusteredError(f"{msg} Run .fit() first.")
        return

    def _check_evaluate_arguments(self, clustering_n_minus_1: Optional["TypicalPeriodsClustering"]):
        if self._n_cluster > 1:
            if clustering_n_minus_1 is None:
                raise EvaluationError(
                    "When evaluating a clustering result with more than one cluster, a clustering "
                    "result with one fewer number of clusters must be passed as an argument."
                )
        return

    def _evaluate_measures(self, clustering_n_minus_1: Optional["TypicalPeriodsClustering"]):
        avg_sq_err = AverageSquaredError(self._tso).value(self._typical_periods, self._labels)
        avg_inter_cluster_dist = AverageInterClusterDistance(self._tso).value(self._typical_periods, self._labels)
        ratio_obs_exp_sq_err = self._ratio_obs_exp_sq_err_and_alpha(avg_sq_err, clustering_n_minus_1)

        self._measures = PerformanceMeasures(avg_sq_err, avg_inter_cluster_dist, ratio_obs_exp_sq_err)
        return

    def _ratio_obs_exp_sq_err_and_alpha(
        self, average_squared_error: float, clustering_n_minus_1: "TypicalPeriodsClustering"
    ) -> float:

        ratio_obs_exp_sq_err_ind = RatioObservedExpectedSquaredError(self._tso)
        if self._n_cluster == 1:
            ratio_obs_exp_sq_err = ratio_obs_exp_sq_err_ind.value(self._typical_periods, average_squared_error)
        elif self._n_cluster == 2:
            ratio_obs_exp_sq_err = ratio_obs_exp_sq_err_ind.value(
                self._typical_periods, average_squared_error, clustering_n_minus_1.average_squared_error
            )
        else:
            ratio_obs_exp_sq_err = ratio_obs_exp_sq_err_ind.value(
                self._typical_periods,
                average_squared_error,
                clustering_n_minus_1.average_squared_error,
                clustering_n_minus_1.alpha,
            )
        self._alpha = ratio_obs_exp_sq_err_ind.alpha
        return ratio_obs_exp_sq_err

    def _evaluate_indicators(self):
        profile_dev = ProfileDeviation(self._tso).value(self._typical_periods, self._labels)
        avg_val_dev = AverageValueDeviation(self._tso).value(self._typical_periods, self._labels)
        load_duration_curve_err = LoadDurationCurveError(self._tso).value(self._typical_periods, self._labels)
        max_load_duration_curve_diff = MaximumLoadDurationCurveDifference(self._tso).value(
            self._typical_periods, self._labels
        )
        periods_err_over_threshold = PeriodsErrorOverThreshold(self._tso, self._gamma).value(
            self._typical_periods, self._labels
        )

        self._indicators = PerformanceIndicators(
            profile_dev, avg_val_dev, load_duration_curve_err, max_load_duration_curve_diff, periods_err_over_threshold
        )
        return

    @property
    def typical_periods(self) -> TypicalPeriods:
        """Get the typical periods determined through the clustering.\
        """
        self._check_is_clustered("Typical periods have not been created yet.")
        return self._typical_periods

    @property
    def labels(self) -> npt.NDArray[Any]:
        """Get the labels that assign the original observations to the typical periods.\
        """
        self._check_is_clustered("Labels have not been created yet.")
        return self._labels

    @property
    def measures(self) -> PerformanceMeasures:
        """Get the performance measures of the typical period clustering result.\
        """
        self._check_is_evaluated("Measures have not been calculated yet.")
        return self._measures

    @property
    def indicators(self) -> PerformanceIndicators:
        """Get the performance indicators of the typical period clustering result.\
        """
        self._check_is_evaluated("Indicators have not been calculated yet.")
        return self._indicators

    @property
    def average_squared_error(self) -> float:
        """Get the average squared error of the typical period clustering result.\
        """
        return self.measures.average_squared_error

    @property
    def alpha(self) -> float:
        """Get the alpha value of the performance measure.\
        """
        self._check_is_evaluated("Alpha has not been calculated yet.")
        return self._alpha

    def _check_is_evaluated(self, msg: str):
        if not self._is_clustered:
            raise NotEvaluatedError(f"{msg} Run .evaluate() first.")
        return
