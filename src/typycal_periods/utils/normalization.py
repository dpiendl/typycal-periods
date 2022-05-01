from typing import Any
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt

from typycal_periods.metrics.exceptions import AttributesError
from typycal_periods.metrics.exceptions import NotFittedError
from typycal_periods.tp_types.time_series import TimeSeriesObservationsBaseClass


class TimeSeriesNormalizer:
    """Normalizer for time series to even the influence of attributes with different units. Provides methods to scale
    the attributes of time series to the same mean with optionally different standard deviations to weight some
    attributes more than others in the subsequent clustering.
    """

    def __init__(self):
        """Constructor method."""
        self._mean_original = None
        # This value is chosen to prevent negative numbers in the scaled time series, even though this should not be an
        # issue.
        self._mean_target: Union[float, npt.NDArray[Any]] = 0.0

        self._std_original: Union[float, npt.NDArray[Any]] = 1.0
        self._std_target: Union[float, Tuple[float], npt.NDArray[Any]] = 1.0

        self._is_fitted = False
        return

    def fit(self, time_series: TimeSeriesObservationsBaseClass):
        """Fit the normalizer to the mean and standard deviation of the time series.

        :param time_series: The time series used to calculate the mean and standard deviation.
        """

        self._mean_original = np.mean(time_series.observations, axis=(0, 1))
        self._std_original = np.std(time_series.observations, axis=(0, 1))
        self._is_fitted = True
        return

    def normalize(
        self,
        time_series: TimeSeriesObservationsBaseClass,
        std_target: Union[float, Tuple[float], npt.NDArray[Any]] = 1.0,
    ):
        """Scale the time series using the fitted mean, fitted standard deviation and target standard deviation with
        the following formula:

            time_series = (time_series - fitted_mean) * (target_std / fitted_std) + target_mean,

        using a target_mean of 100.0 and the supplied target standard deviation.
        Before calling this method, .fit(...) must be called.

        :param time_series: The time series to scale to the target standard deviation and mean.
        :param std_target: The target standard deviation. Can be a scalar that is used for all attributes or an array
            with an individual standard deviation per attribute. Default is 1.0.
        """
        self._assert_is_fitted()
        self._validate_arguments(time_series, std_target)
        self._std_target = std_target

        time_series.observations = time_series.observations - self._mean_original
        time_series.observations /= self._std_original  # scale data so new std is at the target std
        time_series.observations *= self._std_target
        time_series.observations += self._mean_target  # shift data, so new mean is at the target mean
        return

    def fit_normalize(
        self, time_series: TimeSeriesObservationsBaseClass, std_target: Union[float, Tuple[Any], npt.NDArray[Any]] = 1.0
    ):
        """Calls the methods .fit(...) and .scale(...) subsequently in one step. Refer to the respective documentation
        for details.

        Transforms the time series into a time series with the supplied target standard deviation and a mean of 100.

        :param time_series: The time series to scale to the target standard deviation and mean and
            to fit the normalizer.
        :param std_target: The target standard deviation. Can be a scalar that is used for all attributes or an array
            with an individual standard deviation per attribute. Default is 1.0.
        """
        self.fit(time_series)
        self.normalize(time_series, std_target)
        return

    def revert_normalization(self, time_series: TimeSeriesObservationsBaseClass):
        """Reverts a time_series to its original mean and standard deviation.

        :param time_series: The time series to revert to the original standard deviation and mean.
        """
        self._assert_is_fitted()
        time_series.observations = (
            time_series.observations - self._mean_target
        ) / self._std_target * self._std_original + self._mean_original
        return

    def _assert_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("Normalizer must be fitted before scaling a time series. Call .fit(..) first.")
        return

    @staticmethod
    def _validate_arguments(
        time_series: TimeSeriesObservationsBaseClass, std: Union[float, Tuple[Any], npt.NDArray[Any]]
    ):
        std = np.asarray(std)
        if np.ndim(std) == 0:
            if std < 0:
                raise ValueError("The value of std must be non-negative.")
        else:
            if any(std < 0):
                raise ValueError("The elements in std must be non-negative.")
            if len(std) != time_series.n_attributes:
                raise AttributesError(
                    "Mismatch between number of attributes in time_series and elements in std."
                    f"There are {time_series.n_attributes} attributes and {len(std)} elements."
                )

        return
