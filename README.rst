.. image:: https://travis-ci.com/smartcrop/smartcrop.py.svg?branch=master
    :target: https://travis-ci.com/smartcrop/smartcrop.py

smartcrop.py
============

smartcrop implementation in Python.

smartcrop finds good crops for arbitrary images and crop sizes, based on Jonas Wagner's `smartcrop.js`_.

.. _`smartcrop.js`: https://github.com/jwagner/smartcrop.js

.. image:: https://i.gyazo.com/c602d20e025e58f5b15180cd9a262814.jpg
    :width: 50%

.. image:: https://i.gyazo.com/5fbc9026202f54b13938de621562ed3d.jpg
    :width: 25%

.. image:: https://i.gyazo.com/88ee22ca9e1dd7e9eba7ea96db084e5e.jpg
    :width: 50%

Requirements
------------

* PIL or Pillow
* numpy for `smartcrop_numpy.py`

Installation
------------

.. code-block:: sh

    pip3 install smartcrop
    pip3 install numpy  # if you want to use smartcrop_numpy.py

or directly from GitHub:

.. code-block:: sh

    pip install -e git+git://github.com/hhatto/smartcrop.py.git@master#egg=smartcrop

Usage
-----

Use the basic command-line tool:

.. code-block:: sh

    $ smartcroppy --help
    usage: smartcroppy [-h] [--debug] [--width WIDTH] [--height HEIGHT]
                       INPUT_FILE OUTPUT_FILE

    positional arguments:
      INPUT_FILE       input image file
      OUTPUT_FILE      output image file

    optional arguments:
      -h, --help       show this help message and exit
      --debug          debug mode
      --width WIDTH    crop width
      --height HEIGHT  crop height

Processing an image:

.. code-block:: sh

  smartcroppy --width 300 --height 300 tests/images/business-work-1.jpg output.jpg

Or use the module it in your code (this is a really basic example):

.. code-block:: python

    import json
    import sys

    import smartcrop  # or smartcrop_numpy
    from PIL import Image

    image = Image.open(sys.argv[1])

    sc = smartcrop.SmartCrop()
    result = sc.crop(image, 100, 100)
    print(json.dumps(result, indent=2))

License
-------

MIT
