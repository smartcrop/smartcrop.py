from __future__ import division

import argparse
import copy
import json
import math
import sys

from PIL import Image, ImageDraw

DEFAULTS = {
    'width': 0,
    'height': 0,
    'aspect': 0,
    'crop_width': 0,
    'crop_height': 0,
    'detail_weight': 0.2,
    'skin_color': [0.78, 0.57, 0.44],
    'skin_bias': 0.01,
    'skin_brightness_min': 0.2,
    'skin_brightness_max': 1.0,
    'skin_threshold': 0.8,
    'skin_weight': 1.8,
    'saturation_brightness_min': 0.05,
    'saturation_brightness_max': 0.9,
    'saturation_threshold': 0.4,
    'saturation_bias': 0.2,
    'saturation_weight': 0.3,
    # step * minscale rounded down to the next power of two should be good
    'score_down_sample': 8,
    'step': 8,
    'scale_step': 0.1,
    'min_scale': 0.9,
    'max_scale': 1.0,
    'edge_radius': 0.4,
    'edge_weight': -20.0,
    'outside_importance': -0.5,
    'rule_of_thirds': True,
    'prescale': True,
    'debug': False
}


def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = ((x + 2 / 3) % 2 * 0.5 - 0.5) * 16
    return max(1 - x * x, 0)


def cie(r, g, b):
    return 0.5126 * b + 0.7152 * g + 0.0722 * r


def saturation(r, g, b):
    maximum = max(r, g, b)
    minumum = min(r, g, b)
    if maximum == minumum:
        return 0
    s = (maximum + minumum) / 255
    d = (maximum - minumum) / 255
    return d / (2 - d) if s > 1 else d / s


class SmartCrop(object):

    def __init__(self, **options):
        self.options = options or DEFAULTS.copy()

    def crop(self, image, **options):
        if options['aspect']:
            options['width'] = options['aspect']
            options['height'] = 1

        scale = 1
        if options['width'] and options['height']:
            scale = min(image.size[0] / options['width'], image.size[1] / options['height'])
            options['crop_width'] = int(math.floor(options['width'] * scale))
            options['crop_height'] = int(math.floor(options['height'] * scale))
            # img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
            # don't set minscale smaller than 1/scale
            # -> don't pick crops that need upscaling
            options['min_scale'] = min(
                options['max_scale'] or SmartCrop.DEFAULTS.max_scale,
                max(1 / scale, (options['min_scale'] or SmartCrop.DEFAULTS.min_scale)))

        prescale = 1
        if options['width'] and options['height']:
            if options['prescale']:
                prescale = 1 / scale / options['min_scale']
                if prescale < 1:
                    image.thumbnail(
                        (int(image.size[0] * prescale), int(image.size[1] * prescale)),
                        Image.ANTIALIAS)
                    self.options['crop_width'] = int(math.floor(options['crop_width'] * prescale))
                    self.options['crop_height'] = int(math.floor(options['crop_height'] * prescale))
                else:
                    prescale = 1

        result = self.analyse(image)
        for i in range(len(result['crops'])):
            crop = result['crops'][i]
            crop['x'] = int(math.floor(crop['x'] / prescale))
            crop['y'] = int(math.floor(crop['y'] / prescale))
            crop['width'] = int(math.floor(crop['width'] / prescale))
            crop['height'] = int(math.floor(crop['height'] / prescale))
            result['crops'][i] = crop
        return result

    def skin_color(self, r, g, b):
        skin_r, skin_g, skin_b = self.options['skin_color']
        mag = math.sqrt(r * r + g * g + b * b)
        if mag == 0:
            rd = -skin_r
            gd = -skin_g
            bd = -skin_b
        else:
            rd = r / mag - skin_r
            gd = g / mag - skin_g
            bd = b / mag - skin_b
        d = math.sqrt(rd * rd + gd * gd + bd * bd)
        return 1 - d

    def analyse(self, image):
        result = {}
        output_image = Image.new("RGB", image.size, (0, 0, 0))
        output_image = self.detect_edge(image, output_image)
        output_image = self.detect_skin(image, output_image)
        output_image = self.detect_saturation(image, output_image)

        score_output_image = output_image.copy()
        score_output_image.thumbnail(
            (
                int(math.ceil(image.size[0] / self.options['score_down_sample'])),
                int(math.ceil(image.size[1] / self.options['score_down_sample']))
            ),
            Image.ANTIALIAS)

        top_crop = None
        top_score = -sys.maxsize

        crops = self.crops(image)
        for crop in crops:
            crop['score'] = self.score(score_output_image, crop)
            if crop['score']['total'] > top_score:
                top_crop = crop
                top_score = crop['score']['total']

        result['crops'] = crops
        result['top_crop'] = top_crop

        if self.options['debug'] and top_crop:
            debug_output = copy.copy(output_image)
            debug_pixels = debug_output.getdata()
            debug_image = Image.new(
                'RGBA',
                (
                    int(math.floor(top_crop['width'])),
                    int(math.floor(top_crop['height']))
                ),
                (255, 0, 0, 25)
            )
            ImageDraw.Draw(debug_image).rectangle(
                ((0, 0), (top_crop['width'], top_crop['height'])),
                outline=(255, 0, 0))

            for y in range(output_image.size[1]):        # height
                for x in range(output_image.size[0]):    # width
                    p = y * output_image.size[0] + x
                    importance = self.importance(top_crop, x, y)
                    if importance > 0:
                        debug_pixels.putpixel(
                            (x, y),
                            (
                                debug_pixels[p][0],
                                int(debug_pixels[p][1] + importance * 32),
                                debug_pixels[p][2]
                            ))
                    if importance < 0:
                        debug_pixels.putpixel(
                            (x, y),
                            (
                                int(debug_pixels[p][0] + importance * -64),
                                debug_pixels[p][1],
                                debug_pixels[p][2]
                            ))
            debug_output.paste(debug_image, (top_crop['x'], top_crop['y']), debug_image.split()[3])
            debug_output.save('debug.jpg')
        return result

    def detect_edge(self, source_image, target_image):
        source_data = source_image.getdata()
        width, height = source_image.size
        for y in range(height):
            for x in range(width):
                p = y * width + x
                lightness = 0
                if x == 0 or x >= width - 1 or y == 0 or y >= height - 1:
                    lightness = cie(*source_data[p])
                else:
                    lightness = (
                        cie(*source_data[p]) * 4 - cie(*source_data[p - width]) -
                        cie(*source_data[p - 1]) - cie(*source_data[p + 1]) - cie(*source_data[p + width]))
                target_image.putpixel((x, y), (source_data[p][0], int(lightness), source_data[p][2]))
        return target_image

    def detect_skin(self, source_image, target_image):
        source_data = source_image.getdata()
        target_data = target_image.getdata()
        width, height = source_image.size

        brightness_max = self.options['skin_brightness_max']
        brightness_min = self.options['skin_brightness_min']
        threshold = self.options['skin_threshold']

        for y in range(height):
            for x in range(width):
                p = y * width + x
                skin = self.skin_color(source_data[p][0], source_data[p][1], source_data[p][2])
                lightness = cie(source_data[p][0], source_data[p][1], source_data[p][2]) / 255
                if skin > threshold and lightness >= brightness_min and lightness <= brightness_max:
                    target_image.putpixel(
                        (x, y),
                        (
                            int((skin - threshold) * (255 / (1 - threshold))),
                            target_data[p][1],
                            target_data[p][2]
                        ))
                else:
                    target_image.putpixel((x, y), (0, target_data[p][1], target_data[p][2]))
        return target_image

    def detect_saturation(self, source_image, target_image):
        source_data = source_image.getdata()
        target_data = target_image.getdata()
        width, height = source_image.size

        threshold = self.options['saturation_threshold']
        brightness_max = self.options['saturation_brightness_max']
        brightness_min = self.options['saturation_brightness_min']

        for y in range(height):
            for x in range(width):
                p = y * width + x
                lightness = cie(source_data[p][0], source_data[p][1], source_data[p][2]) / 255
                sat = saturation(source_data[p][0], source_data[p][1], source_data[p][2])
                if sat > threshold and lightness >= brightness_min and lightness <= brightness_max:
                    target_image.putpixel(
                        (x, y),
                        (
                            target_data[p][0],
                            target_data[p][1],
                            int((sat - threshold) * (255 / (1 - threshold)))
                        ))
                else:
                    target_image.putpixel((x, y), (target_data[p][0], target_data[p][1], 0))
        return target_image

    def crops(self, image):
        crops = []
        width, height = image.size
        minDimension = min(width, height)
        crop_width = self.options['crop_width'] or minDimension
        crop_height = self.options['crop_height'] or minDimension
        scales = [
            i / 100 for i in range(
                int(self.options['max_scale'] * 100),
                int((self.options['min_scale'] - self.options['scale_step']) * 100),
                -int(self.options['scale_step'] * 100))
        ]
        for scale in scales:
            for y in range(0, height, self.options['step']):
                if not (y + crop_height * scale <= height):
                    break
                for x in range(0, width, self.options['step']):
                    if not (x + crop_width * scale <= width):
                        break
                    crops.append({
                        'x': x,
                        'y': y,
                        'width': crop_width * scale,
                        'height': crop_height * scale,
                    })
        return crops

    def score(self, target_image, crop_image):
        score = {
            'detail': 0,
            'saturation': 0,
            'skin': 0,
            'total': 0,
        }
        target_data = target_image.getdata()
        target_width, target_height = target_image.size

        down_sample = self.options['score_down_sample']
        inv_down_sample = 1 / down_sample
        target_width_down_sample = target_width * down_sample
        target_height_down_sample = target_height * down_sample

        for y in range(0, target_height_down_sample, down_sample):
            for x in range(0, target_width_down_sample, down_sample):
                p = int(
                    math.floor(y * inv_down_sample) * target_width +
                    math.floor(x * inv_down_sample)
                )
                importance = self.importance(crop_image, x, y)
                detail = target_data[p][1] / 255
                score['skin'] += (
                    target_data[p][0] / 255 *
                    (detail + self.options['skin_bias']) *
                    importance
                )
                score['detail'] += detail * importance
                score['saturation'] += (
                    target_data[p][2] / 255 *
                    (detail + self.options['saturation_bias']) *
                    importance
                )
        score['total'] = (
            score['detail'] * self.options['detail_weight'] +
            score['skin'] * self.options['skin_weight'] +
            score['saturation'] * self.options['saturation_weight']
        ) / (crop_image['width'] * crop_image['height'])
        return score

    def importance(self, crop, x, y):
        if (
            crop['x'] > x or x >= crop['x'] + crop['width'] or
            crop['y'] > y or y >= crop['y'] + crop['height']
        ):
            return self.options['outside_importance']

        x = (x - crop['x']) / crop['width']
        y = (y - crop['y']) / crop['height']
        px, py = abs(0.5 - x) * 2, abs(0.5 - y) * 2

        # distance from edge
        dx = max(px - 1 + self.options['edge_radius'], 0)
        dy = max(py - 1 + self.options['edge_radius'], 0)
        d = (dx * dx + dy * dy) * self.options['edge_weight']
        s = 1.41 - math.sqrt(px * px + py * py)

        if self.options['rule_of_thirds']:
            s += (max(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py))

        return s + d


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='INPUT_FILE', help='input image file')
    parser.add_argument('outputfile', metavar='OUTPUT_FILE', help='output image file')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug mode')
    parser.add_argument('--width', dest='width', type=int, default=100, help='crop width')
    parser.add_argument('--height', dest='height', type=int, default=100, help='crop height')
    return parser.parse_args()


def main():
    options = parse_argument()
    crop_options = DEFAULTS.copy()
    crop_options.update({
        'debug': options.debug,
        'width': 100,
        'height': int(options.height / options.width * 100)
    })

    image = Image.open(options.inputfile)
    if image.mode != 'RGB' and image.mode != 'RGBA':
        sys.stderr.write("{1} convert from mode='{0}' to mode='RGB'\n".format(
            image.mode, options.inputfile))
        new_image = Image.new("RGB", image.size)
        new_image.paste(image)
        image = new_image

    result = SmartCrop().crop(image, **crop_options)
    if options.debug:
        print(json.dumps(result))
    box = (
        result['top_crop']['x'],
        result['top_crop']['y'],
        result['top_crop']['width'] + result['top_crop']['x'],
        result['top_crop']['height'] + result['top_crop']['y']
    )
    image = Image.open(options.inputfile)
    image2 = image.crop(box)
    image2.thumbnail((options.width, options.height), Image.ANTIALIAS)
    image2.save(options.outputfile, 'JPEG', quality=90)


if __name__ == '__main__':
    main()
