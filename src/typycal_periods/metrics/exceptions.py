class TypeValidationError(TypeError):
    pass


class ValueValidationError(ValueError):
    pass


class NotClusteredError(RuntimeError):
    pass


class NotEvaluatedError(RuntimeError):
    pass


class NotFittedError(RuntimeError):
    pass


class EvaluationError(RuntimeError):
    pass


class LengthError(ValueError):
    pass


class DimensionError(ValueError):
    pass


class ArrayValuesError(ValueError):
    pass


class IndicatorError(ValueError):
    pass


class TooFewObservationsError(ValueError):
    pass


class AttributesError(ValueError):
    pass
