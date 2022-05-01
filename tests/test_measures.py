import numpy as np
import pytest

from typycal_periods.metrics.measures import AverageInterClusterDistance
from typycal_periods.metrics.measures import AverageSquaredError
from typycal_periods.metrics.measures import IndicatorError
from typycal_periods.metrics.measures import RatioObservedExpectedSquaredError
from typycal_periods.tp_types.time_series import TimeSeriesObservations


def test_avg_squared_error(observations, typical_periods, labels):
    avg_sq_err = AverageSquaredError(observations)
    value = avg_sq_err.value(typical_periods, labels)
    assert value == pytest.approx(10.5)


def test_avg_inter_cluster_dist(observations, typical_periods, labels):
    avg_inter_cluster_dist = AverageInterClusterDistance(observations)
    value = avg_inter_cluster_dist.value(typical_periods, labels)
    assert value == pytest.approx(60.75)


def test_roese_ese(observations, typical_periods):
    roese = RatioObservedExpectedSquaredError(observations)
    value = roese.value(typical_periods, 10.5, 12, 0.8125)
    assert value == pytest.approx(28 / 13)


@pytest.mark.xfail(raises=IndicatorError)
def test_roese_ese_argument_check(observations, typical_periods):
    roese = RatioObservedExpectedSquaredError(observations)
    roese.value(typical_periods, 8)


def test_roese_alpha(observations, typical_periods, typical_periods_1, typical_periods_3):
    not_used = 1.0
    roese = RatioObservedExpectedSquaredError(observations)
    assert roese.alpha is None
    roese.value(typical_periods_1, not_used)
    assert roese.alpha is None
    roese.value(typical_periods, not_used, not_used)
    assert roese.alpha == pytest.approx(0.8125)
    roese.value(typical_periods_3, not_used, not_used, alpha_n_minus_1=0.5)
    assert roese.alpha == pytest.approx(7 / 12)
    roese.value(typical_periods_3, not_used, 0, alpha_n_minus_1=0.5)
    assert roese.alpha == pytest.approx(1)


@pytest.mark.xfail(raises=IndicatorError)
def test_roese_alpha_one_cluster(observations, typical_periods_1):
    roese = RatioObservedExpectedSquaredError(observations)
    roese._calc_alpha(typical_periods_1, None)


@pytest.mark.xfail(raises=IndicatorError)
def test_roese_alpha_one_value(typical_periods):
    observations = TimeSeriesObservations(np.array([[[1], [2]]]))
    roese = RatioObservedExpectedSquaredError(observations)
    roese._calc_alpha(typical_periods, None)


@pytest.mark.xfail(raises=IndicatorError)
def test_roese_alpha_three_cluster_no_alpha_prev(observations, typical_periods_3):
    roese = RatioObservedExpectedSquaredError(observations)
    roese._calc_alpha(typical_periods_3, None)
