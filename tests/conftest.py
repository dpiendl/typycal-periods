import numpy as np
import pytest

from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


@pytest.fixture
def observations():
    observations_ = TimeSeriesObservations(
        np.array([[[1, -5], [4, -5]], [[1, -7], [2, -7]], [[4, -1], [-3, -3]], [[2, 3], [-6, -2]]])
    )
    return observations_


@pytest.fixture
def observations_one_attr():
    observations_ = TimeSeriesObservations(np.array([[[1], [4]], [[1], [2]], [[4], [-3]], [[2], [-6]]]))
    return observations_


@pytest.fixture
def typical_periods():
    typical_periods_ = TypicalPeriods(np.array([[[1, -6], [3, -6]], [[3, 1], [-4.5, -2.5]]]))
    return typical_periods_


@pytest.fixture
def typical_periods_1():
    typical_periods_ = TypicalPeriods(np.array([[[1, -6], [3, -6]]]))
    return typical_periods_


@pytest.fixture
def typical_periods_3():
    typical_periods_ = TypicalPeriods(np.array([[[1, -6], [3, -6]], [[3, 1], [-4.5, -2.5]], [[2, -10], [10, 0]]]))
    return typical_periods_


@pytest.fixture
def labels():
    labels_ = np.array([0, 0, 1, 1])
    return labels_
