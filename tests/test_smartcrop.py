import math
import os
from typing import Final

import numpy as np
import pytest
from PIL import Image, ImageDraw
from smartcrop import RectangleBoost, SmartCrop


def load_image(name):
    here = os.path.abspath(os.path.dirname(__file__))
    img = Image.open(os.path.join(here, 'images', name))
    return img


def transparent_image(mode, hidden_color):
    """Two opaque subjects on a transparent background of `hidden_color`"""
    img = Image.new('P', (800, 400))
    img.putpalette((*hidden_color, 240, 235, 225, 28, 26, 32))
    img.info['transparency'] = 0
    draw = ImageDraw.Draw(img)
    draw.ellipse([120, 130, 380, 270], fill=1)
    draw.ellipse([600, 130, 740, 270], fill=2)
    return img if mode == 'P' else img.convert('RGBA').convert(mode)


@pytest.mark.parametrize('image, crop', [
    ('business-work-1.jpg', (41, 0, 1193, 1152)),
    ('nature-1.jpg', (705, 235, 3642, 3172)),
    ('travel-1.jpg', (52, 52, 1372, 1372)),
    ('orientation.jpg', (972, 0, 3969, 2997))
])
def test_square_thumbs(image, crop):
    cropper = SmartCrop()

    img = load_image(image)
    ret = cropper.crop(img.copy(), 200, 200)

    box = (ret['top_crop']['x'],
           ret['top_crop']['y'],
           ret['top_crop']['width'] + ret['top_crop']['x'],
           ret['top_crop']['height'] + ret['top_crop']['y'])

    print(box)

    if box != crop:
        img = img.crop(box)
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        img.save('thumb.jpg')

    assert box == crop

BAD_BOUNDS_PREFIX = "Bad scale bounds!\n  Expected: 0 < min_scale ≤ max_scale ≤ 1\n  Received: "

@pytest.mark.parametrize("num_scale_steps, step, min_scale, max_scale, expected_error", [
        # num_scale_steps validation
        (2.5, 8, 0.9, 1.0, "num_scale_steps should be an integer! Got: float"),
        ("2", 8, 0.9, 1.0, "num_scale_steps should be an integer! Got: str"),
        (None, 8, 0.9, 1.0, "num_scale_steps should be an integer! Got: NoneType"),
        (0, 8, 0.9, 1.0, "num_scale_steps must be at least 1! Got: 0"),
        (-3, 8, 0.9, 1.0, "num_scale_steps must be at least 1! Got: -3"),

        # step validation
        (2, 8.5, 0.9, 1.0, "step should be an integer! Got: float"),
        (2, "8", 0.9, 1.0, "step should be an integer! Got: str"),
        (2, None, 0.9, 1.0, "step should be an integer! Got: NoneType"),

        # scale bounds validation
        (2, 8, 0, 1.0, BAD_BOUNDS_PREFIX + "0 !         0 ≤       1.0 ≤ 1"),
        (2, 8, -0.1, 1.0, BAD_BOUNDS_PREFIX + "0 !      -0.1 ≤       1.0 ≤ 1"),
        (2, 8, 0.95, 0.9, BAD_BOUNDS_PREFIX + "0 <      0.95 !       0.9 ≤ 1"),
        (2, 8, 0.9, 1.1, BAD_BOUNDS_PREFIX + "0 <       0.9 ≤       1.1 ! 1"),
        (2, 8, 0.5, 1.5, BAD_BOUNDS_PREFIX + "0 <       0.5 ≤       1.5 ! 1"),
        (2, 8, 0, 1.1, BAD_BOUNDS_PREFIX + "0 !         0 ≤       1.1 ! 1"),

        # Edge cases that should NOT raise errors
        (1, 8, 0.9, 1.0, None),              # valid: single scale step
        (2, 8, 0.9, 0.9, None),              # valid: equal scales
        (2, 8, 0.001, 1.0, None),            # valid: very small min_scale
        (5, 16, 0.5, 0.8, None),             # valid: all parameters correct
    ])
def test_crops_validation(num_scale_steps, step, min_scale, max_scale, expected_error):
    """Test validation errors for the crops method"""
    cropper = SmartCrop()

    class MockImage:  #pylint:disable=too-few-public-methods
        size = (5, 5)

    crop_args = {
        "image": MockImage,
        "crop_width": 4,
        "crop_height": 4,
        "num_scale_steps": num_scale_steps,
        "max_scale": max_scale,
        "min_scale":  min_scale,
        "step":  step
    }

    if expected_error is None:
        # This should not raise an error but you never know
        try:
            cropper.crops(**crop_args)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError raised for valid inputs: {e}")
    else:
        with pytest.raises(ValueError) as exc_info:
            cropper.crops(**crop_args)

        error_msg = str(exc_info.value)
        assert expected_error in error_msg


def test_crop_under_one_pixel_raises():
    """A ratio the image holds only at 0 pixels high is refused, not scored as an empty crop"""
    with pytest.raises(ValueError, match=r'must be at least 1! Got: 200x0'):
        SmartCrop().crop(Image.new('RGB', (200, 200)), 1200, 5)


@pytest.mark.filterwarnings('error')
def test_thin_crop_scores_its_content():
    """A crop thinner than a feature cell is scored on one row of cells and lands on the subject"""
    image = Image.new('RGB', (1000, 700), (128, 128, 128))
    image.paste((220, 30, 30), (200, 300, 800, 360))

    result = SmartCrop().crop(image, 100, 6)

    assert all(math.isfinite(crop['score']) for crop in result['crops'])
    assert result['top_crop']['y'] <= 330 < result['top_crop']['y'] + result['top_crop']['height']


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_composite_ignores_hidden_color(mode):
    """Test that visually identical images get the same crops"""
    cropper = SmartCrop(alpha_strategy='composite')

    results = [
        cropper.analyse(transparent_image(mode, hidden_color), 200, 200)['crops']
        for hidden_color in [(0, 0, 0), (255, 255, 255), (255, 0, 0)]
    ]

    assert results[0] == results[1] == results[2]


def test_alpha_fill_is_the_composited_background():
    """Test that transparency is flattened onto alpha_fill"""
    composited = SmartCrop(alpha_strategy='composite', alpha_fill=(255, 0, 0)).analyse(
        transparent_image('RGBA', (0, 0, 0)), 200, 200
    )
    discarded = SmartCrop().analyse(transparent_image('RGBA', (255, 0, 0)), 200, 200)

    assert composited['crops'] == discarded['crops']


def test_unknown_alpha_strategy_raises():
    """Test that a misspelled strategy is refused rather than discarding"""
    cropper = SmartCrop(alpha_strategy='compsite')  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Got: 'compsite'"):
        cropper.analyse(transparent_image('RGBA', (0, 0, 0)), 200, 200)


def test_discard_reads_the_stored_color_when_prescaled():
    """Prescaling keeps the color stored under transparent pixels, which 'discard' scores"""
    image = transparent_image('RGBA', (255, 0, 0))

    prescaled = SmartCrop().crop(image, 200, 200)

    assert prescaled['crops'] == SmartCrop().crop(image.convert('RGB'), 200, 200)['crops']


def test_palette_image_is_prescaled_as_rgb():
    """A palette image is prescaled with the filter of its RGB conversion, not decimated"""
    columns = np.zeros((800, 1600), dtype=np.uint8)
    columns[:, :800:2] = 1
    image = Image.frombytes('P', (1600, 800), columns.tobytes())
    image.putpalette((0, 0, 0, 255, 255, 255))

    prescaled = SmartCrop().crop(image, 200, 200)

    assert prescaled['crops'] == SmartCrop().crop(image.convert('RGB'), 200, 200)['crops']


LEFT_SUBJECT: Final[RectangleBoost] = RectangleBoost(x=120, y=130, width=260, height=140)


def test_prepare_boost_stays_in_range():
    """Hard edges do not ring: each cell is the mean of the pixels it covers"""
    boost_map = np.zeros((400, 400), dtype=np.float32)
    boost_map[100:203, 100:203] = 1

    prepared = SmartCrop.prepare_boost(boost_map, (50, 50))

    assert prepared.min() == 0.0
    assert prepared.max() == 1.0
    assert np.allclose(prepared, boost_map.reshape((50, 8, 50, 8)).mean(axis=(1, 3)))


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_non_finite_boost_raises(value):
    """A NaN or infinite weight is refused: it would make every score NaN or infinite"""
    boost = np.zeros((400, 400), dtype=np.float32)
    boost[0, 0] = value

    with pytest.raises(ValueError, match='finite'):
        SmartCrop().analyse(Image.new('RGB', (400, 400)), 100, 100, boost=boost)


def test_alpha_channel_as_boost():
    """An 8-bit map, such as an alpha channel, counts as its value / 255"""
    image = transparent_image('RGBA', (0, 0, 0))
    alpha = image.getchannel('A')
    cropper = SmartCrop(alpha_strategy='composite')

    from_alpha = cropper.crop(image, 200, 200, boost=alpha)
    from_floats = cropper.crop(image, 200, 200, boost=np.asarray(alpha, dtype=np.float32) / 255)

    assert from_alpha['crops'] == from_floats['crops']


def test_boost_weight_scales_the_boost():
    image = transparent_image('RGBA', (0, 0, 0))
    boost = image | LEFT_SUBJECT

    weighted = SmartCrop(boost_weight=2.0).crop(image, 200, 200, boost=boost)
    doubled = SmartCrop().crop(image, 200, 200, boost=boost * 2)

    assert weighted['crops'] == doubled['crops']


@pytest.mark.parametrize('to_boost', [
    lambda image: image | LEFT_SUBJECT,
    lambda image: np.array([[1, 0]], dtype=np.float32),
    lambda image: image.getchannel('A'),
], ids=['full-size', 'one-cell-per-half', 'alpha'])
def test_boost_moves_the_crop(to_boost):
    """A boost on the pale subject wins over the dark one, at any map resolution"""
    image = transparent_image('RGBA', (0, 0, 0))
    cropper = SmartCrop(alpha_strategy='composite')

    unboosted = cropper.crop(image, 200, 200)['top_crop']
    boosted = cropper.crop(image, 200, 200, boost=to_boost(image))['top_crop']

    def get_center(crop):
        return crop['x'] + crop['width'] // 2

    assert get_center(boosted) < image.width // 2 < get_center(unboosted)
