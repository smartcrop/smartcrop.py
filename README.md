[![image](https://badge.fury.io/py/smartcrop.png)](https://badge.fury.io/py/smartcrop)
[![image](https://travis-ci.com/smartcrop/smartcrop.py.svg?branch=master)](https://travis-ci.com/smartcrop/smartcrop.py)

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
pip install -e git+git://github.com/hhatto/smartcrop.py.git@master#egg=smartcrop
```

## Usage

Use the basic command-line tool:

``` sh
$ smartcroppy --help
usage: smartcroppy [-h] [--debug-file DEBUG_FILE] [--width WIDTH] [--height HEIGHT] INPUT_FILE OUTPUT_FILE

positional arguments:
  INPUT_FILE            Input image file
  OUTPUT_FILE           Output image file

options:
  -h, --help            show this help message and exit
  --debug-file DEBUG_FILE
                        Debugging image file
  --width WIDTH         Crop width
  --height HEIGHT       Crop height
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

## Testing

Install dependencies for testing, then call `pytest`:
``` sh
pip3 install smartcrop[test]
pytest
```

## License

MIT
