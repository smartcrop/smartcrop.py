#!/usr/bin/env python

import sys

import smartcrop
from PIL import Image

image = Image.open(sys.argv[1])

sc = smartcrop.SmartCrop()
result = sc.crop(image, 100, 100)
print(result)
