import numpy as np
import pytest

from typycal_periods.utils.normalization import AttributesError
from typycal_periods.utils.normalization import NotFittedError
from typycal_periods.utils.normalization import TimeSeriesNormalizer


def test_scale(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations)

    for i in range(observations.n_attributes):
        assert np.mean(observations.get(None, None, i)) == pytest.approx(0)
        assert np.std(observations.get(None, None, i)) == pytest.approx(1)


def test_scale_revert(observations):
    normalizer = TimeSeriesNormalizer()
    mean = np.mean(observations.observations, axis=(0, 1))
    std = np.std(observations.observations, axis=(0, 1))
    normalizer.fit_normalize(observations)
    assert all(mean != np.mean(observations.observations, axis=(0, 1)))
    assert all(std != np.std(observations.observations, axis=(0, 1)))
    normalizer.revert_normalization(observations)
    assert np.mean(observations.observations, axis=(0, 1)) == pytest.approx(mean)
    assert np.std(observations.observations, axis=(0, 1)) == pytest.approx(std)


def test_scale_custom_std(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations, std_target=2)

    for i in range(observations.n_attributes):
        assert np.mean(observations.get(None, None, i)) == pytest.approx(0)
        assert np.std(observations.get(None, None, i)) == pytest.approx(2)


def test_scale_custom_stds(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations, std_target=(1, 2))

    for i in range(observations.n_attributes):
        assert np.mean(observations.get(None, None, i)) == pytest.approx(0)
        assert np.std(observations.get(None, None, i)) == pytest.approx(i + 1)


def test_scale_one_attr(observations_one_attr):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations_one_attr, std_target=3)

    assert np.mean(observations_one_attr.get()) == pytest.approx(0)
    assert np.std(observations_one_attr.get()) == pytest.approx(3)


@pytest.mark.xfail(raises=AttributesError)
def test_scale_invalid_args_1(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations, std_target=(1, 2, 3))


@pytest.mark.xfail(raises=ValueError)
def test_scale_invalid_args_2(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations, std_target=(-1, 2))


@pytest.mark.xfail(raises=ValueError)
def test_scale_invalid_args_3(observations_one_attr):
    normalizer = TimeSeriesNormalizer()
    normalizer.fit_normalize(observations_one_attr, std_target=-1)


@pytest.mark.xfail(raises=NotFittedError)
def test_scale_not_fitted(observations):
    normalizer = TimeSeriesNormalizer()
    normalizer.normalize(observations)
