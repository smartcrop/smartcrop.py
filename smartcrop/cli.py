import argparse
import json
import sys

from PIL import Image, ImageOps

from .library import SmartCrop


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('inputfile', metavar='INPUT_FILE', help='Input image file')
    arg('outputfile', metavar='OUTPUT_FILE', help='Output image file')
    arg('--debug-file', metavar='DEBUG_FILE', help='Debugging image file')
    arg('--width', type=int, default=100, help='Crop width')
    arg('--height', type=int, default=100, help='Crop height')
    return parser.parse_args()


def main() -> None:
    options = parse_argument()

    image = Image.open(options.inputfile)

    # Apply orientation from EXIF metadata
    image = image = ImageOps.exif_transpose(image)

    # Ensure image is in RGB (convert it otherwise)
    if image.mode != 'RGB':
        sys.stderr.write(f"{image.mode} convert from mode='{options.inputfile}' to mode='RGB'\n")
        image = image.convert('RGB')

    cropper = SmartCrop()
    result = cropper.crop(image, width=100, height=int(options.height / options.width * 100))

    box = (
        result['top_crop']['x'],
        result['top_crop']['y'],
        result['top_crop']['width'] + result['top_crop']['x'],
        result['top_crop']['height'] + result['top_crop']['y']
    )

    if options.debug_file:
        analyse_image = result.pop('analyse_image')
        cropper.debug_crop(analyse_image, result['top_crop'], image.size).save(options.debug_file)
        print(json.dumps(result))

    cropped_image = image.crop(box)
    cropped_image.thumbnail((options.width, options.height), Image.Resampling.LANCZOS)
    cropped_image.save(options.outputfile, 'JPEG', quality=90)
