import dataclasses

import numpy as np
import pytest

from typycal_periods.tp_types.data_classes import PerformanceIndicators
from typycal_periods.tp_types.data_classes import PerformanceMeasures
from typycal_periods.tp_types.data_classes import TypeValidationError
from typycal_periods.tp_types.data_classes import TypicalPeriodsSelectionConfig
from typycal_periods.tp_types.data_classes import ValueValidationError


@pytest.fixture
def measure():
    return PerformanceMeasures(1.0, 2.0, 3.0)


@pytest.fixture
def indicator_1():
    return PerformanceIndicators(
        np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8]), np.array([9, 10])
    )


@pytest.fixture
def indicator_2():
    return PerformanceIndicators(
        np.array([11, 12]), np.array([13, 14]), np.array([15, 16]), np.array([17, 18]), np.array([19, 20])
    )


def test_measures(measure):
    assert measure.average_squared_error == 1.0
    assert measure.average_inter_cluster_distance == 2.0
    assert measure.ratio_observed_expected_squared_error == 3.0
    return


def test_measure_eq(measure):
    measure2 = PerformanceMeasures(1.0, 2.0, 3.0)
    assert measure == measure2
    return


@pytest.mark.xfail(raises=dataclasses.FrozenInstanceError)
def test_measure_not_mutable(measure):
    measure.average_squared_error = 2


def test_indicators_add(indicator_1, indicator_2):
    assert indicator_1 + indicator_2 == PerformanceIndicators(
        np.array([12, 14]), np.array([16, 18]), np.array([20, 22]), np.array([24, 26]), np.array([28, 30])
    )


def test_indicator_sub(indicator_1, indicator_2):
    assert indicator_2 - indicator_1 == PerformanceIndicators(
        np.array([10, 10]), np.array([10, 10]), np.array([10, 10]), np.array([10, 10]), np.array([10, 10])
    )


def test_indicator_mul(indicator_1, indicator_2):
    assert indicator_1 * indicator_1 == PerformanceIndicators(
        np.array([1, 4]), np.array([9, 16]), np.array([25, 36]), np.array([49, 64]), np.array([81, 100])
    )


def test_indicator_div(indicator_1):
    assert indicator_1 / indicator_1 == PerformanceIndicators(
        np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])
    )


@pytest.mark.xfail(raises=ValueError)
def test_indicator_arithmetic_with_scalar(indicator_1):
    _ = indicator_1 * 1


def test_indicator_comp(indicator_1, indicator_2):
    assert not indicator_1 < indicator_1
    assert indicator_1 <= indicator_1
    assert indicator_1 == indicator_1
    assert indicator_1 >= indicator_1
    assert not indicator_1 > indicator_1

    assert indicator_1 != indicator_2
    assert indicator_1 < indicator_2
    assert indicator_2 > indicator_1
    assert not indicator_1 >= indicator_2
    assert not indicator_2 <= indicator_1


def test_indicator_comp2(indicator_1):
    assert indicator_1 > 0
    assert not indicator_1 > 1
    assert not indicator_1 < 1
    assert indicator_1 < 11
    assert indicator_1 <= 10
    assert not indicator_1 < 10
    assert not indicator_1 <= 9
    assert indicator_1 >= 1
    assert not indicator_1 >= 2
    assert PerformanceIndicators(*[np.array([1, 1])] * 5) == 1
    assert not PerformanceIndicators(*[np.array([1, 1])] * 5) == 2


@pytest.mark.parametrize("gamma,pareto_error", [(0, 0.9), (1, 2)])
def test_tp_config(gamma, pareto_error):
    tp = TypicalPeriodsSelectionConfig(gamma, pareto_error)
    assert tp.gamma == gamma
    assert tp.pareto_error == pareto_error


@pytest.mark.xfail(raises=TypeValidationError)
def test_tp_invalid_type():
    TypicalPeriodsSelectionConfig(gamma=[1])


@pytest.mark.xfail(raises=ValueValidationError)
@pytest.mark.parametrize("gamma", (-0.1,))
def test_tp_invalid_gamma(gamma):
    TypicalPeriodsSelectionConfig(gamma=gamma)


@pytest.mark.xfail(raises=ValueValidationError)
def test_tp_invalid_pareto():
    TypicalPeriodsSelectionConfig(pareto_error=-0.1)
