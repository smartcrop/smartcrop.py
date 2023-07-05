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
    ('nature-1.jpg', (705, 235, 3639, 3169)),
    ('travel-1.jpg', (52, 52, 1370, 1370)),
    ('orientation.jpg', (972, 216, 3669, 2913))
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
