import os

import pytest
from PIL import Image
from smartcrop import SmartCrop


def load_image(name):
    here = os.path.abspath(os.path.dirname(__file__))
    img = Image.open(os.path.join(here, 'images', name))
    return img


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
