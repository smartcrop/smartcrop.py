.. image:: https://travis-ci.com/smartcrop/smartcrop.py.svg?branch=master
    :target: https://travis-ci.com/smartcrop/smartcrop.py

smartcrop.py
============

smartcrop implementation in Python

smartcrop finds good crops for arbitrary images and crop sizes, based on Jonas Wagner's `smartcrop.js`_

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


Installation
------------

.. code-block:: sh

    pip2 install smartcrop
    pip3 install smartcrop

or directly from GitHub:

.. code-block:: sh

    pip install -e git+git://github.com/hhatto/smartcrop.py.git@master#egg=smartcrop

Usage
-----

command-line tool

.. code-block:: sh

    smartcroppy FILE

use module

.. code-block:: python

    import json
    import sys

    import smartcrop
    from PIL import Image

    image = Image.open(sys.argv[1])

    sc = smartcrop.SmartCrop()
    result = sc.crop(image, 100, 100)
    print(json.dumps(result, indent=2))

smartcrop.py is slower than `smartcrop.js`_

.. code-block:: sh

    $ identify images/t.jpg
    images/t.jpg JPEG 3200x2403 3200x2403+0+0 8-bit DirectClass 2.066MB 0.000u 0:00.000
    $ time smartcrop --width 300 --height 300 images/t.jpg
    smartcrop --width 300 --height 300 images/t.jpg  0.30s user 0.11s system 100% cpu 0.414 total
    $ time smartcroppy --width 300 --height 300 images/t.jpg
    smartcroppy --width 300 --height 300 images/t.jpg  3.74s user 0.31s system 99% cpu 4.051 total

License
-------

MIT
