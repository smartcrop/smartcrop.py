import sys
import json
from PIL import Image
import smartcrop

sc = smartcrop.SmartCrop()
img = Image.open(sys.argv[1])
ret = sc.crop(img, width=100, height=100)
print(json.dumps(ret, indent=2))
