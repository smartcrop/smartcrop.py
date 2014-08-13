smartcrop.py
============
smartcrop implementation in Python

smartcrop finds good crops for arbitrary images and crop sizes, based on Jonas Wagner's `smartcrop.js`_

.. _`smartcrop.js`: https://github.com/jwagner/smartcrop.js

.. image:: https://dl.dropboxusercontent.com/u/26471561/img/smartcroppy/bw.jpg
    :width: 50%

.. image:: https://dl.dropboxusercontent.com/u/26471561/img/smartcroppy/bw_out.jpg
    :width: 25%

.. image:: https://dl.dropboxusercontent.com/u/26471561/img/smartcroppy/bw_debug.jpg
    :width: 50%

Requirements
------------
* PIL or Pillow


Installation
------------
.. code-block:: sh

    pip install --upgrade git+https://github.com/hhatto/smartcrop.py.git


Usage
-----
command-line tool

.. code-block:: sh

    smartcrop.py FILE

use module

.. code-block:: python

    import sys
    import json
    from PIL import Image
    import smartcrop

    sc = smartcrop.SmartCrop()
    crop_options = smartcrop.DEFAULTS
    crop_options['width'] = 100
    crop_options['height'] = 100

    img = Image.open(sys.argv[1])
    ret = sc.crop(img, crop_options)
    print(json.dumps(ret, indent=2))


smartcrop.py is slower than `smartcrop.js`_

.. code-block:: sh

    $ identify images/t.jpg
    images/t.jpg JPEG 3200x2403 3200x2403+0+0 8-bit DirectClass 2.066MB 0.000u 0:00.000
    $ smartcrop --width 300 --height 300 images/t.jpg
    smartcrop --width 300 --height 300 images/t.jpg  0.30s user 0.11s system 100% cpu 0.414 total
    $ time smartcroppy --width 300 --height 300 images/t.jpg
    smartcroppy --width 300 --height 300 images/t.jpg  3.74s user 0.31s system 99% cpu 4.051 total

License
-------
MIT
