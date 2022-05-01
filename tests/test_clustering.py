import numpy as np
import pytest

from typycal_periods.clustering.cluster import EvaluationError
from typycal_periods.clustering.cluster import NotClusteredError
from typycal_periods.clustering.cluster import NotEvaluatedError
from typycal_periods.clustering.cluster import PerformanceIndicators
from typycal_periods.clustering.cluster import PerformanceMeasures
from typycal_periods.clustering.cluster import TypicalPeriodsClustering


@pytest.fixture
def clustering_1(observations):
    gamma = 0.4
    n_cluster = 1
    clustering = TypicalPeriodsClustering(observations, n_cluster, gamma)
    return clustering


@pytest.fixture
def clustering_2(observations):
    gamma = 0.4
    n_cluster = 2
    clustering = TypicalPeriodsClustering(observations, n_cluster, gamma)
    return clustering


def test_clustering_fit(clustering_2):
    clustering_2.fit()
    assert (clustering_2.typical_periods.observations == np.array([[[1, -6], [3, -6]], [[3, 1], [-4.5, -2.5]]])).all()


def test_clustering_eval_1(clustering_1):
    clustering_1.fit()
    clustering_1.evaluate()


def test_clustering_eval_2(clustering_2):
    clustering_2.fit()
    clustering_1 = type("obj", (object,), {"average_squared_error": 12})
    clustering_2.evaluate(clustering_1)
    assert clustering_2.measures == PerformanceMeasures(10.5, 60.75, 28 / 13)
    assert clustering_2.indicators == PerformanceIndicators(
        np.array([0.63245553, 0.15713484]),
        np.array([1.30304798, 1.29431769]),
        np.array([0, 0]),
        np.array([0.25, 0.66666667]),
        np.array([0, 2]),
    )


def test_clustering_eval_3(observations):
    gamma = 0.4
    clustering = TypicalPeriodsClustering(observations, 3, gamma)
    clustering.fit()
    clustering_2 = type("obj", (object,), {"average_squared_error": 10.5, "alpha": 0.8125})
    clustering.evaluate(clustering_2)


def test_clustering_getters(clustering_2):
    clustering_2.fit()
    clustering_1 = type("obj", (object,), {"average_squared_error": 12})
    clustering_2.evaluate(clustering_1)
    assert clustering_2.average_squared_error == 10.5
    assert clustering_2.alpha == 0.8125


@pytest.mark.xfail(raises=NotClusteredError)
def test_clustering_check_is_clustered(clustering_2):
    clustering_2.evaluate()


@pytest.mark.xfail(raises=EvaluationError)
def test_clustering_evaluate_arguments_supply_prev_result(clustering_2):
    clustering_2.fit()
    clustering_2.evaluate()


@pytest.mark.xfail(raises=NotEvaluatedError)
def test_clustering_evaluate_arguments_eval_prev_result(clustering_1, clustering_2):
    clustering_2.fit()
    clustering_2.evaluate(clustering_1)
