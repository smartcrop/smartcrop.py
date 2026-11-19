[![image](https://badge.fury.io/py/smartcrop.png)](https://badge.fury.io/py/smartcrop)

[![Python package](https://github.com/smartcrop/smartcrop.py/actions/workflows/python-package.yml/badge.svg)](https://github.com/smartcrop/smartcrop.py/actions/workflows/python-package.yml)

# smartcrop.py

smartcrop implementation in Python.

smartcrop finds good crops for arbitrary images and crop sizes, based on
Jonas Wagner\'s [smartcrop.js](https://github.com/jwagner/smartcrop.js).

![image](https://i.gyazo.com/c602d20e025e58f5b15180cd9a262814.jpg)
![image](https://i.gyazo.com/5fbc9026202f54b13938de621562ed3d.jpg)
![image](https://i.gyazo.com/88ee22ca9e1dd7e9eba7ea96db084e5e.jpg)

## Requirements

Python requirements are defined in [pyproject.toml](pyproject.toml).


## Installation

``` sh
pip3 install -U pip setuptools wheel # optional but recommended
pip3 install smartcrop
```

or directly from GitHub:

``` sh
pip3 install -U pip setuptools wheel # optional but recommended
pip install -e git+git://github.com/smartcrop/smartcrop.py.git@master#egg=smartcrop
```

## Usage

Use the basic command-line tool:

``` sh
$ smartcroppy --help
usage: smartcroppy [-h] [--debug-file DEBUG_FILE] [--width WIDTH] [--height HEIGHT] INPUT_FILE OUTPUT_FILE

positional arguments:
  INPUT_FILE            Input image file (BMP, GIF, JPEG, PNG, TIFF, WEBP)
  OUTPUT_FILE           Output image file, in the format of its extension

options:
  -h, --help            show this help message and exit
  --debug-file DEBUG_FILE
                        Debugging image file
  --width WIDTH         Crop width, at most
  --height HEIGHT       Crop height, at most
```

Processing an image:

``` sh
smartcroppy --width 300 --height 300 tests/images/business-work-1.jpg output.jpg --debug-file debug.jpg
```

Or use the module it in your code (this is a really basic example, see [examples/](examples/) and [smartcrop/cli.py](smartcrop/cli.py) for inspiration):

``` python
import json
import sys

import smartcrop
from PIL import Image

image = Image.open(sys.argv[1])
cropper = smartcrop.SmartCrop()
result = cropper.crop(image, 100, 100)
print(json.dumps(result, indent=2))
```

### Boost

A boost biases crop selection toward regions of interest. It is a weight map, a NumPy array or a
Pillow image, added to the image features before the crops are scored. Any library can produce
it: a saliency or segmentation model, a face detector's boxes drawn with numpy, `ImageDraw`,
`skimage.draw` or OpenCV.

* **Resolution:** any, with the image's aspect ratio. The map is averaged down to the features grid
  (`score_down_sample`), and follows the image through prescaling.
* **Values:** the same units as the other features, scaled by `SmartCrop.boost_weight` (default
  `1.0`). A float array or a mode `F` image is taken as-is; any other image mode, such as an alpha
  channel, is scaled to 0..1.

The `RectangleBoost` helper draws rectangles in image pixels. Boosts add up with `+` or `sum()`, and
overlapping weights accumulate. `image | boost` draws them into a weight map sized to the image:

``` python
import smartcrop
from PIL import Image

image = Image.open('photo.jpg')
face = smartcrop.RectangleBoost(x=120, y=40, width=80, height=80, weight=1.0)
eyes = smartcrop.RectangleBoost(x=130, y=60, width=60, height=20, weight=0.5)

cropper = smartcrop.SmartCrop()
result = cropper.crop(image, 300, 300, boost=image | face + eyes)
print(result['top_crop'])
```

Subclass `smartcrop.Boost` and implement `paint()` to draw other shapes.

The alpha channel of a transparent image boosts its opaque content:

``` python
cropper = smartcrop.SmartCrop(alpha_strategy='composite')
result = cropper.crop(image, 300, 300, boost=image.getchannel('A'))
```

## Testing

Install dependencies for testing, then call `pytest`:
``` sh
pip3 install smartcrop[test]
pytest
```

## License

MIT
