from abc import ABCMeta
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt

from typycal_periods.metrics.exceptions import ArrayValuesError
from typycal_periods.metrics.exceptions import DimensionError
from typycal_periods.metrics.exceptions import LengthError


class TimeSeriesObservationsBaseClass:
    """Base class for time series observations.\
    """

    __metaclass__ = ABCMeta

    def __init__(self, observations: npt.NDArray[Any], attribute_names: Optional[Tuple[str, ...]] = None):
        """Constructor method.

        :param observations: A numpy.ndarray object of the shape
            [number of observations, length of observation, number of attributes]
        """
        self._check_observations(observations)

        self._observations: npt.NDArray[Any] = self._copy_observations(observations)

        self._attribute_names: Tuple[str, ...] = self._init_attribute_names(attribute_names)
        return

    @staticmethod
    def _check_observations(observations: npt.NDArray[Any]):
        if not len(observations.shape) == 3:
            raise DimensionError(
                f"Argument observations has incorrect dimension of {len(observations.shape)}, " f"should be 3."
            )
        return

    @staticmethod
    def _copy_observations(observations: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return observations.copy()

    def _init_attribute_names(self, attribute_names: Optional[Tuple[str, ...]] = None):
        if attribute_names is None:
            attribute_names_ = list()
            for i in range(self.n_attributes):
                attribute_names_.append(f"Attribute {i+1}")
            attribute_names = tuple(attribute_names_)

        if len(attribute_names) != self.n_attributes:
            raise LengthError(
                f"Length of attributes names must be equal to the number of attributes. "
                f"Was {len(attribute_names)}, should be {self.n_attributes}."
            )

        return attribute_names

    @property
    def observations(self) -> npt.NDArray[Any]:
        """Get the observations.\
        """
        return self._observations

    @observations.setter
    def observations(self, value):
        self._observations = value
        return

    @property
    def n_observations(self) -> int:
        """Get the number of observations.\
        """
        return self.observations.shape[0]

    @property
    def len_observation(self) -> int:
        """Get the length of one observation.\
        """
        return self.observations.shape[1]

    @property
    def n_attributes(self) -> int:
        """Get the number of attributes.\
        """
        return self.observations.shape[2]

    @property
    def attribute_names(self) -> Tuple[str, ...]:
        """Get the names of the attributes.\
        """
        return self._attribute_names

    def get(
        self,
        observation: Optional[Union[int, slice]] = None,
        step: Optional[Union[int, slice]] = None,
        attribute: Optional[Union[int, slice]] = None,
    ) -> Union[float, npt.NDArray[Any]]:
        """Get the time series observations by index by using integers, slices or None, where None gets all indexes
        along the given axis (same as colon ':').

        :param observation: The index or indices for the observations to get.
        :param step: The index or indices for the time steps to get.
        :param attribute: The index or indices for the attributes to get.
        :return: Scalar or numpy-array of the subset of observations.
        """
        idx_observation = slice(None) if observation is None else observation
        idx_step = slice(None) if step is None else step
        idx_attribute = slice(None) if attribute is None else attribute
        return self.observations[idx_observation, idx_step, idx_attribute]


class TimeSeriesObservations(TimeSeriesObservationsBaseClass):
    """A time series object containing multiple observations with the same length and number of attributes.\
    """

    pass


class TypicalPeriods(TimeSeriesObservationsBaseClass):
    """A time series object containing multiple typical periods with the same length and number of attributes as well
    as weights determining the importance of each typical period.
    """

    def __init__(
        self,
        typical_periods: npt.NDArray[Any],
        attribute_names: Optional[Tuple[str, ...]] = None,
        weights: Optional[npt.NDArray[Any]] = None,
    ):
        """Constructor method.

        :param typical_periods: A numpy.ndarray object of the shape
            [number of typical periods, length of typical period, number of attributes].
        :param weights: An array of the weight for each typical period. Must have a single dimension and a length equal
            to the number of typical periods. The sum of all elements must be equal to 1. Defaults to an even
            distribution.
        """
        super().__init__(typical_periods, attribute_names)
        self.weights: npt.NDArray[Any] = self._init_weights(weights)
        return

    def _init_weights(self, weights: Optional[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        if weights is None:
            weights = np.ones(self.n_typical_periods) / self.n_typical_periods
        return weights

    @property
    def typical_periods(self) -> npt.NDArray[Any]:
        """Get the typical periods. Identical to method 'observations'.\
        """
        return self._observations.copy()

    @property
    def n_typical_periods(self) -> int:
        """Get the number of typical periods. Identical to method 'n_observations'.\
        """
        return self.n_observations

    @property
    def weights(self) -> npt.NDArray[Any]:
        """Get the weights for the typical periods.\
        """
        return self._weights

    @weights.setter
    def weights(self, weights: npt.NDArray[Any]):
        """Set the weights for the typical periods. Must have a single dimension and a length equal to the number of
        typical periods. The sum of all elements must be equal to 1.
        """
        if weights.ndim != 1:
            raise DimensionError(f"Argument weights must have one dimension, was {weights.ndim}.")
        if len(weights) != self.n_typical_periods:
            raise LengthError(
                f"Argument weights must have a length equal to "
                f"the number of typical periods ({self.n_typical_periods}), was {len(weights)}."
            )
        if not np.isclose(sum(weights), 1):
            raise ArrayValuesError(f"The sum of all elements of argument weights must be 1, was {sum(weights)}.")
        self._weights = weights
        return
