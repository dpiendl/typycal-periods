import numpy as np
import pytest

from typycal_periods.tp_types.time_series import ArrayValuesError
from typycal_periods.tp_types.time_series import DimensionError
from typycal_periods.tp_types.time_series import LengthError
from typycal_periods.tp_types.time_series import TimeSeriesObservations
from typycal_periods.tp_types.time_series import TypicalPeriods


@pytest.fixture
def observations():
    n_observations = 365
    len_observation = 24
    n_attributes = 5
    data = np.arange(n_observations * len_observation * n_attributes)
    data_reshaped = data.reshape((n_observations, len_observation, n_attributes))
    return data_reshaped


@pytest.fixture
def ts_observations(observations):
    ts_observation = TimeSeriesObservations(observations)
    return ts_observation


def test_observations(ts_observations):
    n_observations = 365
    len_observation = 24
    n_attributes = 5
    data = np.arange(n_observations * len_observation * n_attributes)
    data_reshaped = data.reshape((n_observations, len_observation, n_attributes))
    assert (ts_observations.observations == data_reshaped).all()


def test_observations_set(ts_observations):
    ts_observations.observations = ts_observations.observations


def test_observations_2(ts_observations):
    assert ts_observations.observations.shape == (365, 24, 5)


def test_n_observations(ts_observations):
    assert ts_observations.n_observations == 365


def test_len_observation(ts_observations):
    assert ts_observations.len_observation == 24


def test_n_attributes(ts_observations):
    assert ts_observations.n_attributes == 5


def test_copy_works():
    ts_data = np.arange(1).reshape((1, 1, 1))
    ts_observations = TimeSeriesObservations(ts_data)
    ts_data[0, 0, 0] = 1
    assert ts_observations.observations[0, 0, 0] == 0


def test_get_correct_value(ts_observations):
    assert ts_observations.get(0, 0, 0) == 0
    assert ts_observations.get(0, 0, 1) == 1
    assert ts_observations.get(1, 0, 0) == 24 * 5
    assert ts_observations.get(1, 1, 0) == 24 * 5 + 5
    assert ts_observations.get(5, 10, 1) == 651
    assert ts_observations.get(-1, -1, -1) == 43799
    assert np.all(ts_observations.get(0, None, None) == np.arange(24 * 5).reshape(24, 5))
    assert np.all(ts_observations.get(-1, None, None) == np.arange(364 * 24 * 5, 365 * 24 * 5).reshape(24, 5))
    assert ts_observations.get(None, 0, None)[0, 0] == 0
    assert ts_observations.get(None, 0, None)[0, 4] == 4
    assert ts_observations.get(None, 0, None)[1, 2] == 122
    assert ts_observations.get(None, None, 0)[1, 0] == 120
    assert ts_observations.get(None, None, 1)[1, 1] == 126
    assert ts_observations.get(None, None, -1)[-1, -1] == 43799
    assert np.all(ts_observations.get(None, None, None) == ts_observations.observations)


def test_get_correct_shape(ts_observations):
    assert np.isscalar(ts_observations.get(0, 0, 0))
    assert ts_observations.get(None, 0, 0).ndim == 1
    assert ts_observations.get(0, None, 0).ndim == 1
    assert ts_observations.get(0, 0, None).ndim == 1
    assert ts_observations.get(None, 0, None).ndim == 2
    assert ts_observations.get(None, None, 0).ndim == 2
    assert ts_observations.get(0, None, None).ndim == 2
    assert ts_observations.get(None, None, None).ndim == 3


def test_attribute_names(observations):
    tp = TimeSeriesObservations(observations, ("Attr 1", "Attr 2", "Attr 3", "Attr 4", "Attr 5"))
    assert tp.attribute_names[4] == "Attr 5"
    assert tp.attribute_names[1] == "Attr 2"


@pytest.mark.xfail(raises=LengthError)
def test_attribute_names_invalid_length(observations):
    TimeSeriesObservations(observations, ("Attr 1", "Attr 2", "Attr 3", "Attr 4"))


def test_typical_periods(observations):
    tp = TypicalPeriods(observations)
    n_observations = 365
    len_observation = 24
    n_attributes = 5
    data = np.arange(n_observations * len_observation * n_attributes)
    data_reshaped = data.reshape((n_observations, len_observation, n_attributes))
    assert (tp.typical_periods == data_reshaped).all()


def test_get_typical_periods(observations):
    tp = TypicalPeriods(observations)
    assert tp.get(None, None, 0)[1, 0] == 120


def test_typical_periods_weights(observations):
    tp = TypicalPeriods(observations)
    assert tp.weights is not None
    random_array = np.random.random(tp.n_typical_periods)
    weights = random_array / sum(random_array)
    tp.weights = weights
    assert all(tp.weights == weights)


@pytest.mark.xfail(raises=DimensionError)
def test_typical_periods_weights_wrong_dim(observations):
    tp = TypicalPeriods(observations)
    tp.weights = np.random.random((tp.n_typical_periods, 2))


@pytest.mark.xfail(raises=LengthError)
def test_typical_periods_weights_wrong_len(observations):
    tp = TypicalPeriods(observations)
    tp.weights = np.random.random(tp.n_typical_periods + 1)


@pytest.mark.xfail(raises=ArrayValuesError)
def test_typical_periods_weights_wrong_values(observations):
    tp = TypicalPeriods(observations)
    tp.weights = np.ones(tp.n_typical_periods)


@pytest.mark.xfail(raises=AttributeError)
def test_not_mutable_2(ts_observations):
    ts_observations.n_observations = None


@pytest.mark.xfail(raises=AttributeError)
def test_not_mutable_3(ts_observations):
    ts_observations.len_observation = None


@pytest.mark.xfail(raises=AttributeError)
def test_not_mutable_4(ts_observations):
    ts_observations.n_attributes = None


@pytest.mark.xfail(raises=DimensionError)
def test_wrong_dimensions():
    ts_data = np.arange(1)
    TimeSeriesObservations(ts_data)


def test_typical_periods_n_cluster(observations):
    typical_periods = TypicalPeriods(observations)
    assert typical_periods.n_typical_periods == 365
