import numpy as np
import pytest

from typycal_periods.metrics.indicators import AverageValueDeviation
from typycal_periods.metrics.indicators import LoadDurationCurveError
from typycal_periods.metrics.indicators import MaximumLoadDurationCurveDifference
from typycal_periods.metrics.indicators import PeriodsErrorOverThreshold
from typycal_periods.metrics.indicators import ProfileDeviation


def test_profile_deviation(observations, typical_periods, labels):
    profile_deviation = ProfileDeviation(observations)
    pd_value = profile_deviation.value(typical_periods, labels)
    pd_expected = np.array([0.63245553, 0.15713484])
    assert pd_value == pytest.approx(pd_expected)
    return


def test_average_value_deviation(observations, typical_periods, labels):
    average_value_deviation = AverageValueDeviation(observations)
    avd_value = average_value_deviation.value(typical_periods, labels)
    avd_expected = np.array([1.30304798, 1.29431769])
    assert avd_value == pytest.approx(avd_expected)
    return


def test_load_duration_curve_error(observations, typical_periods, labels):
    load_duration_curve_error = LoadDurationCurveError(observations)
    ldce_value = load_duration_curve_error.value(typical_periods, labels)
    ldce_expected = np.array([0, 0])
    assert ldce_value == pytest.approx(ldce_expected)
    return


def test_load_duration_curve_difference(observations, typical_periods, labels):
    load_duration_curve_difference = MaximumLoadDurationCurveDifference(observations)
    ldcd_value = load_duration_curve_difference.value(typical_periods, labels)
    ldcd_expected = np.array([0.25, 0.66666667])
    assert ldcd_value == pytest.approx(ldcd_expected)
    return


def test_periods_error_over_threshold(observations, typical_periods, labels):
    gamma = 0.4
    periods_error_over_threshold = PeriodsErrorOverThreshold(observations, gamma)
    peot_value = periods_error_over_threshold.value(typical_periods, labels)
    peot_expected = np.array([0, 2])
    assert peot_value == pytest.approx(peot_expected)
    return


@pytest.mark.xfail(raises=ValueError)
def test_periods_error_over_threshold_gamma_below_range(observations, typical_periods, labels):
    gamma = -0.01
    PeriodsErrorOverThreshold(observations, gamma)
