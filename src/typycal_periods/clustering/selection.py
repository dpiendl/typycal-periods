from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from typycal_periods.clustering.cluster import TypicalPeriodsClustering
from typycal_periods.metrics.exceptions import NotEvaluatedError
from typycal_periods.tp_types.data_classes import TypicalPeriodsSelectionConfig
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


class TypicalPeriodsSelection:
    """This class clusters an original time series observations repeatedly into a different number of typical periods
    and determines the minimum and maximum number of periods with a number of indicators and measures.
    """

    def __init__(
        self,
        time_series_observation: TimeSeriesObservations,
        n_typical_periods_max: int,
        config: TypicalPeriodsSelectionConfig = TypicalPeriodsSelectionConfig(),
    ):
        """Constructor method.

        :param time_series_observation: The original time series observations as a TimeSeriesObservations object.
        :param n_typical_periods_max: The maximum number of clusters that will be used to generate the typical periods.
        :param config: The configuration to calculate indicator values and determine the minimum viable number of
            clusters. Defaults to the default parameters of the TypicalPeriodsSelectionConfig dataclass.
        """
        self._tso: TimeSeriesObservations = time_series_observation
        self._n_cluster_max: int = n_typical_periods_max
        self._config: TypicalPeriodsSelectionConfig = config

        self._typical_periods_clustering: Dict[int, TypicalPeriodsClustering] = self._init_clustering()

        self._ranking: Optional[Dict[int, int]] = None
        self._n_min: Optional[int] = None
        self._n_opt: Optional[int] = None

        self._is_evaluated: bool = False
        return

    def _init_clustering(self) -> Dict[int, TypicalPeriodsClustering]:
        typical_periods_clustering = dict()
        for n_cluster_i in range(1, self._n_cluster_max + 1):
            typical_periods_clustering[n_cluster_i] = TypicalPeriodsClustering(
                self._tso, n_cluster_i, self._config.gamma
            )
        return typical_periods_clustering

    def fit(self):
        """Clusters the time series observations into typical periods. This is repeated for 1 to n_typical_periods_max
        number of typical periods.
        """
        for clustering_i in self._typical_periods_clustering.values():
            clustering_i.fit()
        return

    def evaluate(self):
        """Evaluates the clustering result for each number of typical periods."""
        self._evaluate_each_clustering()
        self._select_n_opt()
        self._is_evaluated = True
        return

    def _evaluate_each_clustering(self):
        clustering_i_prev = None
        for n_cluster_i in range(1, self._n_cluster_max + 1):
            clustering_i = self._typical_periods_clustering[n_cluster_i]
            clustering_i.evaluate(clustering_i_prev)
            clustering_i_prev = clustering_i
        return

    def _select_n_opt(self):
        self._evaluate_ranking()
        self._evaluate_n_min()
        self._evaluate_n_opt(self._ranking, self._n_min)
        return

    def _evaluate_ranking(self):
        ranking = np.empty((3, self._n_cluster_max))
        ranking[0] = self._rank_average_squared_error()
        ranking[1] = self._rank_average_inter_cluster_distance()
        ranking[2] = self._rank_ratio_observed_expected_error()

        self._ranking = {
            n_cluster: rank for n_cluster, rank in zip(range(1, self._n_cluster_max + 1), np.max(ranking, axis=0))
        }
        return

    def _evaluate_n_min(self):

        n_min = self._n_cluster_max
        for n_cluster_i in range(1, self._n_cluster_max):
            these_indicators = self._typical_periods_clustering[n_cluster_i].indicators
            next_indicators = self._typical_periods_clustering[n_cluster_i + 1].indicators
            improvement = next_indicators.improvement_to(these_indicators)

            if improvement <= self._config.pareto_error:
                n_min = n_cluster_i
                break

        self._n_min = n_min
        return

    def _evaluate_n_opt(self, ranking: Dict[int, int], n_min: int):
        ranking_filtered = {n_tp: rank for n_tp, rank in ranking.items() if n_tp >= n_min}
        self._n_opt = min(ranking_filtered, key=ranking_filtered.get)
        return

    def _rank_average_squared_error(self) -> npt.NDArray[Any]:
        array = [tp_i.measures.average_squared_error for _, tp_i in self._typical_periods_clustering.items()]
        array_sorted = rankdata(array, method="min")
        return array_sorted

    def _rank_average_inter_cluster_distance(self) -> npt.NDArray[Any]:
        array = [tp_i.measures.average_inter_cluster_distance for _, tp_i in self._typical_periods_clustering.items()]
        array_sorted = rankdata(array, method="max")
        return array_sorted

    def _rank_ratio_observed_expected_error(self) -> npt.NDArray[Any]:
        array = [
            tp_i.measures.ratio_observed_expected_squared_error for _, tp_i in self._typical_periods_clustering.items()
        ]
        array_sorted = rankdata(array, method="min")
        return array_sorted

    def typical_periods_result(self, n_typical_periods: int) -> TypicalPeriodsClustering:
        """The typical period clustering result for the specified number of periods, including the typical periods,
        measures and indicators.

        :param n_typical_periods: The number of typical periods to fetch the clustering result for.
        :return: The clustering result.
        """
        return self._typical_periods_clustering[n_typical_periods]

    def typical_periods(self, n_typical_periods: int) -> TypicalPeriods:
        """The typical periods for the specified number of periods.

        :param n_typical_periods: The number of typical periods to fetch the typical periods for.
        :return: The typical periods.
        """
        return self.typical_periods_result(n_typical_periods).typical_periods

    @property
    def n_opt(self):
        """The optimal number of typical periods.\
        """
        self._check_is_evaluated()
        return self._n_opt

    @property
    def n_min(self):
        """The minimum viable number of typical periods.\
        """
        self._check_is_evaluated()
        return self._n_min

    @property
    def n_max(self):
        """The maximum number typical periods were created for.\
        """
        return self._n_cluster_max

    def _check_is_evaluated(self):
        if not self._is_evaluated:
            raise NotEvaluatedError(
                "The typical period clustering must first be evaluated before accessing the "
                "minimum and optimal number of typical periods."
            )
        return
