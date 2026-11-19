"""
Shapes that draw themselves into the weight map `SmartCrop` takes as `boost`.

Boosts add up with `+` into one flat `BoostSum`. Any iterable of boosts adds up with
`BoostSum(boosts)`, which draws an all-zero map when it is empty.

**Example usage**

>>> from PIL import Image
>>> image = Image.new('RGB', (4, 2))
>>> top_left = RectangleBoost(x=0, y=0, width=2, height=1)
>>> right = RectangleBoost(x=1, y=0, width=3, height=2, weight=0.5)
>>> (image | top_left + right).tolist()
[[1.0, 1.5, 0.5, 0.5], [0.0, 0.5, 0.5, 0.5]]
>>> (image | BoostSum([])).tolist()
[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PIL import Image


class Boost(ABC):
    """A region of interest that adds weights into a map; boosts add up with `+`."""

    __slots__ = ()

    def __add__(self, other: object) -> BoostSum:
        if not isinstance(other, Boost):
            return NotImplemented
        return BoostSum((*self.terms, *other.terms))

    def __radd__(self, other: object) -> Boost:
        """Return this boost unchanged when added to 0, the start value of `sum()`."""
        if other != 0:
            return NotImplemented
        return self

    def __ror__(self, image: Image.Image) -> np.ndarray:
        """Return this boost as a float32 weight map sized to the image, for `image | boost`."""
        boost_map = np.zeros((image.height, image.width), dtype=np.float32)
        self.paint(boost_map)
        return boost_map

    @property
    def terms(self) -> tuple[Boost, ...]:
        """Return the boosts this one adds up: itself, or the members of a sum."""
        return (self,)

    @abstractmethod
    def paint(self, boost_map: np.ndarray) -> None:
        """Add this boost's weights into `boost_map`, a (height, width) array, in place."""


@dataclass(frozen=True, init=False, slots=True)
class BoostSum(Boost):
    """Boosts added together; their weights accumulate where they overlap."""

    boosts: tuple[Boost, ...]

    def __init__(self, boosts: Iterable[Boost] = ()) -> None:
        object.__setattr__(self, 'boosts', tuple(boosts))

    @override
    @property
    def terms(self) -> tuple[Boost, ...]:
        return self.boosts

    @override
    def paint(self, boost_map: np.ndarray) -> None:
        for boost in self.boosts:
            boost.paint(boost_map)


@dataclass(frozen=True, slots=True)
class RectangleBoost(Boost):
    """A rectangle in image pixels; the part outside the image is ignored."""

    x: int
    y: int
    width: int
    height: int
    weight: float = 1.0

    @override
    def paint(self, boost_map: np.ndarray) -> None:
        # Clamp the bounds at 0: a negative slice bound counts from the far edge.
        rows = slice(max(self.y, 0), max(self.y + self.height, 0))
        columns = slice(max(self.x, 0), max(self.x + self.width, 0))
        boost_map[rows, columns] += self.weight
