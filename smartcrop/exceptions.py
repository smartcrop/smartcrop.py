"""Errors raised on an input `SmartCrop` cannot crop with, each one a `ValueError`."""


# [CONV-PY-ALPHA-ORDER: the base class comes before its subclasses]
class SmartCropError(ValueError):
    """An input `SmartCrop` cannot crop with."""


class AlphaStrategyError(SmartCropError):
    """An `alpha_strategy` other than 'discard' and 'composite'."""


class BoostValueError(SmartCropError):
    """A boost map holding a NaN or infinite value."""


class CropParameterError(SmartCropError):
    """A crop size, scale bound or step `crops()` cannot generate crops with."""


class NoCropError(SmartCropError):
    """An image smaller than every crop size asked for."""
