#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import json
import math
import sys

from PIL import Image, ImageDraw
from PIL.ImageFilter import Kernel


def saturation(r, g, b):
    maximum = max(r, g, b)
    minimum = min(r, g, b)
    if maximum == minimum:
        return 0
    s = (maximum + minimum) / 255
    d = (maximum - minimum) / 255
    return d / (2 - d) if s > 1 else d / s


def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = ((x + 2 / 3) % 2 * 0.5 - 0.5) * 16
    return max(1 - x * x, 0)


class SmartCrop(object):

    DEFAULT_SKIN_COLOR = [0.78, 0.57, 0.44]

    def __init__(
        self,
        detail_weight=0.2,
        edge_radius=0.4,
        edge_weight=-20,
        outside_importance=-0.5,
        rule_of_thirds=True,
        saturation_bias=0.2,
        saturation_brightness_max=0.9,
        saturation_brightness_min=0.05,
        saturation_threshold=0.4,
        saturation_weight=0.3,
        score_down_sample=8,
        skin_bias=0.01,
        skin_brightness_max=1,
        skin_brightness_min=0.2,
        skin_color=None,
        skin_threshold=0.8,
        skin_weight=1.8,
    ):
        self.detail_weight = detail_weight
        self.edge_radius = edge_radius
        self.edge_weight = edge_weight
        self.outside_importance = outside_importance
        self.rule_of_thirds = rule_of_thirds
        self.saturation_bias = saturation_bias
        self.saturation_brightness_max = saturation_brightness_max
        self.saturation_brightness_min = saturation_brightness_min
        self.saturation_threshold = saturation_threshold
        self.saturation_weight = saturation_weight
        self.score_down_sample = score_down_sample
        self.skin_bias = skin_bias
        self.skin_brightness_max = skin_brightness_max
        self.skin_brightness_min = skin_brightness_min
        self.skin_color = skin_color or self.DEFAULT_SKIN_COLOR
        self.skin_threshold = skin_threshold
        self.skin_weight = skin_weight

    def analyse(
        self,
        image,
        crop_width,
        crop_height,
        debug=False,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8
    ):
        """
        Analyze image and return some suggestions of crops (coordinates).
        This implementation / algorithm is really slow for large images.
        Use `crop()` which is pre-scaling the image before analyzing it.
        """
        output_image = Image.new('RGB', image.size, (0, 0, 0))

        self._cie_image = image.convert('L', (0.2126, 0.7152, 0.0722, 0))

        output_image = self.detect_edge(image, output_image)
        output_image = self.detect_skin(image, output_image)
        output_image = self.detect_saturation(image, output_image)

        score_output_image = output_image.copy()
        score_output_image.thumbnail(
            (
                int(math.ceil(image.size[0] / self.score_down_sample)),
                int(math.ceil(image.size[1] / self.score_down_sample))
            ),
            Image.ANTIALIAS)

        top_crop = None
        top_score = -sys.maxsize

        crops = self.crops(
            image,
            crop_width,
            crop_height,
            max_scale=max_scale,
            min_scale=min_scale,
            scale_step=scale_step,
            step=step)

        for crop in crops:
            crop['score'] = self.score(score_output_image, crop)
            if crop['score']['total'] > top_score:
                top_crop = crop
                top_score = crop['score']['total']

        if debug and top_crop:
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

        return {'crops': crops, 'top_crop': top_crop}

    def crop(
        self,
        image,
        width,
        height,
        debug=False,
        prescale=True,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8
    ):
        """Not yet fully cleaned from https://github.com/hhatto/smartcrop.py."""
        scale = min(image.size[0] / width, image.size[1] / height)
        crop_width = int(math.floor(width * scale))
        crop_height = int(math.floor(height * scale))
        # img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
        # don't set minscale smaller than 1/scale
        # -> don't pick crops that need upscaling
        min_scale = min(max_scale, max(1 / scale, min_scale))

        prescale_size = 1
        if prescale:
            prescale_size = 1 / scale / min_scale
            if prescale_size < 1:
                image = image.copy()
                image.thumbnail(
                    (int(image.size[0] * prescale_size), int(image.size[1] * prescale_size)),
                    Image.ANTIALIAS)
                crop_width = int(math.floor(crop_width * prescale_size))
                crop_height = int(math.floor(crop_height * prescale_size))
            else:
                prescale_size = 1

        result = self.analyse(
            image,
            crop_width=crop_width,
            crop_height=crop_height,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
            step=step)

        for i in range(len(result['crops'])):
            crop = result['crops'][i]
            crop['x'] = int(math.floor(crop['x'] / prescale_size))
            crop['y'] = int(math.floor(crop['y'] / prescale_size))
            crop['width'] = int(math.floor(crop['width'] / prescale_size))
            crop['height'] = int(math.floor(crop['height'] / prescale_size))
            result['crops'][i] = crop
        return result

    def crops(
        self,
        image,
        crop_width,
        crop_height,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8
    ):
        image_width, image_height = image.size
        crops = []
        for scale in (
            i / 100 for i in range(
                int(max_scale * 100),
                int((min_scale - scale_step) * 100),
                -int(scale_step * 100))
        ):
            for y in range(0, image_height, step):
                if not (y + crop_height * scale <= image_height):
                    break
                for x in range(0, image_width, step):
                    if not (x + crop_width * scale <= image_width):
                        break
                    crops.append({
                        'x': x,
                        'y': y,
                        'width': crop_width * scale,
                        'height': crop_height * scale,
                    })
        if not crops:
            raise ValueError(locals())
        return crops

    def detect_edge(self, source_image, target_image):
        cie = self._cie_image.convert('L')
        kernel = Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 1)
        edges = cie.filter(kernel)

        r, _, b = target_image.split()
        target_image = Image.merge(target_image.mode, [r, edges, b])

        return target_image

    def detect_saturation(self, source_image, target_image):
        source_data = source_image.getdata()
        target_data = target_image.getdata()
        width, height = source_image.size

        brightness_max = self.saturation_brightness_max
        brightness_min = self.saturation_brightness_min
        threshold = self.saturation_threshold

        cie_data = self._cie_image.getdata()

        for y in range(height):
            for x in range(width):
                p = y * width + x
                lightness = cie_data[p] / 255
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

    def detect_skin(self, source_image, target_image):
        source_data = source_image.getdata()
        target_data = target_image.getdata()
        width, height = source_image.size

        brightness_max = self.skin_brightness_max
        brightness_min = self.skin_brightness_min
        threshold = self.skin_threshold

        cie_data = self._cie_image.getdata()

        for y in range(height):
            for x in range(width):
                p = y * width + x
                skin = self.get_skin_color(source_data[p][0], source_data[p][1], source_data[p][2])
                lightness = cie_data[p] / 255
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

    def get_skin_color(self, r, g, b):
        skin_r, skin_g, skin_b = self.skin_color
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

    def importance(self, crop, x, y):
        if (
            crop['x'] > x or x >= crop['x'] + crop['width'] or
            crop['y'] > y or y >= crop['y'] + crop['height']
        ):
            return self.outside_importance

        x = (x - crop['x']) / crop['width']
        y = (y - crop['y']) / crop['height']
        px, py = abs(0.5 - x) * 2, abs(0.5 - y) * 2

        # distance from edge
        dx = max(px - 1 + self.edge_radius, 0)
        dy = max(py - 1 + self.edge_radius, 0)
        d = (dx * dx + dy * dy) * self.edge_weight
        s = 1.41 - math.sqrt(px * px + py * py)

        if self.rule_of_thirds:
            s += (max(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py))

        return s + d

    def score(self, target_image, crop_image):
        score = {
            'detail': 0,
            'saturation': 0,
            'skin': 0,
            'total': 0,
        }
        target_data = target_image.getdata()
        target_width, target_height = target_image.size

        down_sample = self.score_down_sample
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
                    (detail + self.skin_bias) *
                    importance
                )
                score['detail'] += detail * importance
                score['saturation'] += (
                    target_data[p][2] / 255 *
                    (detail + self.saturation_bias) *
                    importance
                )
        score['total'] = (
            score['detail'] * self.detail_weight +
            score['skin'] * self.skin_weight +
            score['saturation'] * self.saturation_weight
        ) / (crop_image['width'] * crop_image['height'])
        return score


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

    image = Image.open(options.inputfile)
    if image.mode != 'RGB' and image.mode != 'RGBA':
        sys.stderr.write("{1} convert from mode='{0}' to mode='RGB'\n".format(
            image.mode, options.inputfile))
        new_image = Image.new('RGB', image.size)
        new_image.paste(image)
        image = new_image

    result = SmartCrop().crop(
        image,
        width=100,
        height=int(options.height / options.width * 100),
        debug=options.debug)

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
