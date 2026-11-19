import sys

import numpy as np
import pytest
from PIL import Image

from smartcrop import BoostSum, RectangleBoost


def test_rectangle_boost_paints_inside_the_image():
    """Overlapping boosts add up, and the parts outside the image are dropped."""
    image = Image.new('RGB', (6, 2))
    boost = (
        RectangleBoost(x=-2, y=0, width=4, height=1) +
        RectangleBoost(x=1, y=0, width=9, height=5, weight=0.5)
    )

    assert (image | boost).tolist() == [
        [1.0, 1.5, 0.5, 0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]


def test_boost_sum_of_a_list_is_their_addition():
    """A list of boosts adds up through `BoostSum` and `sum()` the same as chained `+`."""
    image = Image.new('RGB', (100, 100))
    boosts = [
        RectangleBoost(x=0, y=0, width=60, height=100),
        RectangleBoost(x=40, y=20, width=60, height=50, weight=0.5),
        RectangleBoost(x=10, y=10, width=10, height=10, weight=2.0),
    ]
    added = image | (boosts[0] + boosts[1] + boosts[2])

    assert np.array_equal(image | BoostSum(boosts), added)
    assert np.array_equal(image | sum(boosts), added)


@pytest.mark.parametrize('boost_sum', [BoostSum(), BoostSum([])], ids=['no-argument', 'empty-list'])
def test_empty_boost_sum_is_a_zero_map(boost_sum):
    """A list of boosts that came out empty draws an all-zero map."""
    assert np.array_equal(Image.new('RGB', (4, 2)) | boost_sum, np.zeros((2, 4)))


def test_boost_sum_keeps_its_boosts():
    """A boost sum built from a one-shot iterable draws the same map every time."""
    image = Image.new('RGB', (4, 2))
    boost_sum = BoostSum(RectangleBoost(x=x, y=0, width=1, height=2) for x in range(2))

    assert np.array_equal(image | boost_sum, image | boost_sum)
    assert boost_sum == BoostSum([RectangleBoost(x=x, y=0, width=1, height=2) for x in range(2)])


def test_sum_of_boosts_stays_flat():
    """Any number of boosts adds up, without nesting one sum per `+` until recursion fails."""
    count = sys.getrecursionlimit() + 1
    boosts = [RectangleBoost(x=x, y=0, width=1, height=1) for x in range(count)]

    assert np.array_equal(Image.new('RGB', (count, 1)) | sum(boosts), np.ones((1, count)))
