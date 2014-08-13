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
