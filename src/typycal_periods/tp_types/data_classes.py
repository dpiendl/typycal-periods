from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import numpy as np
import numpy.typing as npt

from typycal_periods.metrics.exceptions import TypeValidationError
from typycal_periods.metrics.exceptions import ValueValidationError


@dataclass(frozen=True, eq=True)
class PerformanceMeasures:
    """Data class used for storing the values of the performance measures.\
    """

    average_squared_error: float
    average_inter_cluster_distance: float
    ratio_observed_expected_squared_error: float


@dataclass(frozen=True)
class PerformanceIndicators:
    """Data class used for storing the values of the performance indicators.\
    """

    profile_deviation: npt.NDArray[Any]
    average_value_deviation: npt.NDArray[Any]
    load_duration_curve_error: npt.NDArray[Any]
    maximum_load_duration_curve_difference: npt.NDArray[Any]
    periods_over_error_threshold: npt.NDArray[Any]

    def __add__(self, other):
        self._check_arithmetic_arguments(other)
        values = dict()
        for field in fields(self):
            values[field.name] = getattr(self, field.name) + getattr(other, field.name)
        return PerformanceIndicators(**values)

    def __sub__(self, other):
        self._check_arithmetic_arguments(other)
        values = dict()
        for field in fields(self):
            values[field.name] = getattr(self, field.name) - getattr(other, field.name)
        return PerformanceIndicators(**values)

    def __mul__(self, other):
        self._check_arithmetic_arguments(other)
        values = dict()
        for field in fields(self):
            values[field.name] = getattr(self, field.name) * getattr(other, field.name)
        return PerformanceIndicators(**values)

    def __truediv__(self, other):
        self._check_arithmetic_arguments(other)
        values = dict()
        for field in fields(self):
            values[field.name] = getattr(self, field.name) / getattr(other, field.name)
        return PerformanceIndicators(**values)

    @staticmethod
    def _check_arithmetic_arguments(other):
        if not isinstance(other, PerformanceIndicators):
            raise ValueError(
                "Arithmetic operations for instances of PerformanceIndicators are only defined "
                f"with other instances of PerformanceIndicators. Other type was {type(other)}."
            )

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for field_1 in fields(self):
                if not all(np.isclose(getattr(self, field_1.name), other)):
                    return False
        elif isinstance(other, PerformanceIndicators):
            for field_1, field_2 in zip(fields(self), fields(other)):
                if not np.all(np.isclose(getattr(self, field_1.name), getattr(other, field_2.name))):
                    return False
        return True

    def __lt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for field_1 in fields(self):
                if not all(getattr(self, field_1.name) < other):
                    return False
        elif isinstance(other, PerformanceIndicators):
            for field_1, field_2 in zip(fields(self), fields(other)):
                if not np.all(getattr(self, field_1.name) < getattr(other, field_2.name)):
                    return False
        return True

    def __le__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for field_1 in fields(self):
                if not all(np.logical_or(getattr(self, field_1.name) < other, getattr(self, field_1.name) == other)):
                    return False
        elif isinstance(other, PerformanceIndicators):
            for field_1, field_2 in zip(fields(self), fields(other)):
                if not all(
                    np.logical_or(
                        getattr(self, field_1.name) < getattr(other, field_2.name),
                        getattr(self, field_1.name) == getattr(other, field_2.name),
                    )
                ):
                    return False
        return True

    def __gt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for field_1 in fields(self):
                if not all(other < getattr(self, field_1.name)):
                    return False
        elif isinstance(other, PerformanceIndicators):
            for field_1, field_2 in zip(fields(self), fields(other)):
                if not np.all(getattr(other, field_2.name) < getattr(self, field_1.name)):
                    return False
        return True

    def __ge__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for field_1 in fields(self):
                if not all(np.logical_or(other < getattr(self, field_1.name), other == getattr(self, field_1.name))):
                    return False
        elif isinstance(other, PerformanceIndicators):
            for field_1, field_2 in zip(fields(self), fields(other)):
                if not all(
                    np.logical_or(
                        getattr(other, field_2.name) < getattr(self, field_1.name),
                        getattr(other, field_2.name) == getattr(self, field_1.name),
                    )
                ):
                    return False
        return True

    def improvement_to(self, other: "PerformanceIndicators") -> "PerformanceIndicators":
        """Calculates the improvement of the indicator values compared to the other indicator values, taking the other
        indicator values as a baseline: improvement = (this - other)/other. If a field in other is 0, the improvement
        of the field is also set to 0.

        :param other: The other performance indicators to compare to.
        :return: The relative change between this and the other indicator.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = (other - self) / other
        improvement._set_nan_to_zero()
        return improvement

    def _set_nan_to_zero(self):
        for field_1 in fields(self):
            field_value = getattr(self, field_1.name)
            field_value[np.isnan(field_value)] = 0
        return


@dataclass
class TypicalPeriodsSelectionConfig:
    # noinspection PyUnresolvedReferences
    """This is a configuration dataclass for the selection of typical periods.

    :param gamma: Relative error margin for the PeriodsErrorOverThreshold indicator.
        Must be non-negative. Defaults to 0.07.
    :param pareto_error: Relative indicator improvement which determines the minimum viable number of typical
        periods. Must be non-negative. Defaults to 0.2.
    """
    gamma: float = 0.1
    pareto_error: float = 0.2

    def __post_init__(self):
        self._cast_to_correct_type()
        self._validate()
        return

    def _cast_to_correct_type(self):
        self.gamma = self._cast_helper(self.gamma, "gamma")
        self.pareto_error = self._cast_helper(self.pareto_error, "pareto_error")
        return

    @staticmethod
    def _cast_helper(value, name: str):
        try:
            cast_value = float(value)
        except TypeError:
            raise TypeValidationError(f"Cannot cast argument {name} to float. Original type is {type(value)}.")
        return cast_value

    def _validate(self):
        if not 0 <= self.gamma:
            raise ValueValidationError(f"The value of gamma must be non-negative. It was {self.gamma}.")
        if not 0 <= self.pareto_error:
            raise ValueValidationError(f"The value of pareto_error must be non-negative. It was {self.pareto_error}.")
        return
