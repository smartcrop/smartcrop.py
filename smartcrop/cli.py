import argparse
import json
import sys
import warnings
from typing import Final

from PIL import Image, ImageOps, UnidentifiedImageError

from .library import SmartCrop

# The common formats only, keeping Pillow's rarer decoders away from the input.
INPUT_FORMATS: Final[tuple[str, ...]] = ('BMP', 'GIF', 'JPEG', 'PNG', 'TIFF', 'WEBP')


def main() -> None:
    parser = create_parser()
    options = parser.parse_args()

    # An input in another format, or over Pillow's pixel limit, is an argument error.
    try:
        with (
            warnings.catch_warnings(action='error', category=Image.DecompressionBombWarning),
            Image.open(options.inputfile, formats=INPUT_FORMATS) as source,
        ):
            # Apply orientation from EXIF metadata
            image = ImageOps.exif_transpose(source)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
    ) as error:
        parser.error(str(error))

    # Ensure image is in RGB (convert it otherwise)
    if image.mode != 'RGB':
        sys.stderr.write(f"{options.inputfile} convert from mode='{image.mode}' to mode='RGB'\n")
        image = image.convert('RGB')

    cropper = SmartCrop()
    result = cropper.crop(image, width=options.width, height=options.height)

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

    # Shrink to the requested size, and never enlarge a smaller crop.
    if cropped_image.width > options.width:
        cropped_image = cropped_image.resize(
            (options.width, options.height),
            Image.Resampling.LANCZOS,
        )
    cropped_image.save(options.outputfile, quality=90)


# --- Utils ----------------------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('inputfile', metavar='INPUT_FILE', help=f"Input image file ({', '.join(INPUT_FORMATS)})")
    arg(
        'outputfile',
        metavar='OUTPUT_FILE',
        help='Output image file, in the format of its extension',
    )
    arg('--debug-file', metavar='DEBUG_FILE', help='Debugging image file')
    arg('--width', type=parse_size, default=100, help='Crop width, at most')
    arg('--height', type=parse_size, default=100, help='Crop height, at most')
    return parser


def parse_size(value: str) -> int:
    """Return `value` as a size in pixels, raising ArgumentTypeError when it is under 1."""
    if (size := int(value)) < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {size}')
    return size
