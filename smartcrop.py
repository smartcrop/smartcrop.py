#!/usr/bin/env python
import argparse
import copy
import json
import math
import sys

from PIL import Image, ImageDraw, ImageMath
from PIL.ImageFilter import Kernel

import numpy as np

# Laplace kernel for edge detection
LAPLACE_KERNEL = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


def conv2d(a, f):
    """Source: https://stackoverflow.com/a/43087771/1211607 """
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2, writeable=False)
    return np.einsum('ij,ijkl->kl', f, subM)


def edge_detection(image):
    edges = conv2d(np.pad(image, 1, 'edge'), LAPLACE_KERNEL)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return edges


def importance_map(width, height, rule_of_thirds=True, edge_radius=0.4, edge_weight=-20):
    """Draw a importance map

    Regions which are bright are considered important, dark regions unimportant
    """
    lx = np.linspace(0, 1, width)
    ly = np.linspace(0, 1, height)

    # Importance Implementation of smartcrop
    px = np.abs(0.5 - lx) * 2
    py = np.abs(0.5 - ly) * 2
    dx = px - 1 + edge_radius
    dy = py - 1 + edge_radius
    dx[dx < 0] = 0
    dy[dy < 0] = 0

    PX, PY = np.meshgrid(px, py, sparse=True)
    DX, DY = np.meshgrid(dx, dy, sparse=True)

    d = (DX**2 + DY**2) * edge_weight
    s = 1.41 - np.sqrt(PX**2 + PY**2)

    if rule_of_thirds:
        a = ((lx * 2 - 1 / 3) % 2 * 0.5 - 0.5) * 16
        a = 1 - a**2
        a[a < 0] = 0
        b = ((ly * 2 - 1 / 3) % 2 * 0.5 - 0.5) * 16
        b = 1 - b**2
        b[b < 0] = 0

        thirds = (a[:,np.newaxis].T + b[:,np.newaxis])
        thirds += np.flip(thirds)

        # Apply rule of thirds
        t = (s + d + 0.5) * 1.2
        t[t<0] = 0
        s += t * thirds

    return s + d


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

    def analyse(self, image, crop_width, crop_height, debug=False,
                max_scale=1, min_scale=0.8, scale_step=0.1):
        """
        Analyze image and return some suggestions of crops (coordinates).
        This implementation / algorithm is really slow for large images.
        Use `crop()` which is pre-scaling the image before analyzing it.
        """
        # It's not realy grayscale, we use the CIE transformation here 
        self._gray = np.array(image.convert('L', (0.2126, 0.7152, 0.0722, 0))) / 255.0
        self._hsv = image.convert('HSV')

        self.img_edges = edge_detection(self._gray * 255)
        self.detect_skin(image)
        self.detect_saturation(image)

        output_image = Image.merge('RGB', [
            Image.fromarray(self.img_skin),
            Image.fromarray(self.img_edges),
            Image.fromarray(self.img_saturation)
        ])

        # Delete images which are no longer required
        del self.img_skin
        del self.img_edges
        del self.img_saturation

        if debug:
            skin, edge, sat = output_image.split()
            edge.save('debug/out_edge.jpg')
            sat.save('debug/out_sat.jpg')
            skin.save('debug/out_skin.jpg')

        # Scale down image for the scoring
        score_output_image = output_image.copy()
        score_output_image.thumbnail(
            (
                image.size[0] / self.score_down_sample,
                image.size[1] / self.score_down_sample
            ),
            Image.ANTIALIAS)

        if debug:
            score_output_image.save('debug/score_image.jpg')

        # Build the arrays for the scoring
        skin, detail, saturation = score_output_image.split()
        skin, detail, saturation = map(np.array, (skin, detail, saturation))
        skin = skin / 255.0
        detail = detail / 255.0
        saturation = saturation / 255.0

        # Calculate the score for each pixel
        self._img_total_score = (detail * self.detail_weight) +\
            (saturation * (detail + self.saturation_bias) * self.saturation_weight) +\
            (skin * (detail + self.skin_bias) * self.skin_weight)

        crops = []
        top_score = -sys.maxsize
        top_crop = None
        for scale in np.arange(max_scale, min_scale - scale_step, -scale_step):
            # Create a importance map for the given scale
            mp = importance_map(int(crop_width * scale / self.score_down_sample), 
                                int(crop_height * scale / self.score_down_sample), 
                                self.edge_radius, self.edge_weight)
                                
            # Convolve with the score image
            tst = conv2d(self._img_total_score, mp)
            # Extract the location of the maximal score    
            argmax = np.unravel_index(np.argmax(tst), tst.shape)
            # Get the score, adjust the score by the scale
            score = tst[argmax[0],argmax[1]] / scale
            # Build the crop information
            crop = {
                'scale': scale, 
                'score': score,
                'width': crop_width * scale,
                'height': crop_height * scale,
                'x': argmax[1] * scale * self.score_down_sample,
                'y': argmax[0] * scale * self.score_down_sample
            }
            crops.append(crop)
            if top_score < score:
                top_score = score
                top_crop = crop

        return {'crops': crops, 'top_crop': top_crop}

    def crop(self, image, width, height, debug=False, prescale=True,
             max_scale=1, min_scale=0.9, scale_step=0.1):
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

        if debug:
            image.save('debug/prescaled.jpg')

        result = self.analyse(
            image,
            crop_width=crop_width,
            crop_height=crop_height,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
            debug=debug)

        for i in range(len(result['crops'])):
            crop = result['crops'][i]
            crop['x'] = int(crop['x'] / prescale_size)
            crop['y'] = int(crop['y'] / prescale_size)
            crop['width'] = int(crop['width'] / prescale_size)
            crop['height'] = int(crop['height'] / prescale_size)
        return result


    def detect_saturation(self, source_image):
        brightness_max = self.saturation_brightness_max
        brightness_min = self.saturation_brightness_min
        threshold = self.saturation_threshold

        gray = self._gray
        sat = np.array(self._hsv.split()[1]) / 255

        sat = (sat - threshold) * (255 / (1 - threshold))
        mask = (sat < 0) | ~((gray >= brightness_min) & (gray <= brightness_max))
        sat[mask] = 0

        self.img_saturation = sat.astype('uint8')

    def detect_skin(self, source_image):
        brightness_max = self.skin_brightness_max
        brightness_min = self.skin_brightness_min
        threshold = self.skin_threshold

        r, g, b = source_image.split()
        r, g, b = np.array(r, float), np.array(g, float), np.array(b, float)
        mag = np.sqrt(r * r + g * g + b * b)
        rd = np.ones_like(r) * -self.skin_color[0]
        gd = np.ones_like(g) * -self.skin_color[1]        
        bd = np.ones_like(b) * -self.skin_color[2]

        mask = ~(abs(mag) < 1e-6)
        rd[mask] = r[mask] / mag[mask] - self.skin_color[0]
        gd[mask] = g[mask] / mag[mask] - self.skin_color[1]
        bd[mask] = b[mask] / mag[mask] - self.skin_color[2]

        skin = 1 - np.sqrt(rd * rd + gd * gd + bd * bd)
        skinimg = (skin - threshold) * (255 / (1 - threshold))
        mask = (skin > threshold) & (self._gray >= brightness_min) & (self._gray <= brightness_max)
        skinimg[~mask] = 0
        self.img_skin = skinimg.astype('uint8')


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='INPUT_FILE', help='input image file')
    parser.add_argument('outputfile', metavar='OUTPUT_FILE', help='output image file')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug mode')
    parser.add_argument('--width', dest='width', type=int, default=100, help='crop width')
    parser.add_argument('--height', dest='height', type=int, default=100, help='crop height')
    return parser.parse_args()


def main():
    import time
    options = parse_argument()

    image = Image.open(options.inputfile)
    if image.mode != 'RGB' and image.mode != 'RGBA':
        sys.stderr.write("{1} convert from mode='{0}' to mode='RGB'\n".format(
            image.mode, options.inputfile))
        new_image = Image.new('RGB', image.size)
        new_image.paste(image)
        image = new_image

    start = time.time()
    result = SmartCrop().crop(
        image,
        width=options.width,
        height=options.height,
        debug=options.debug,
        min_scale=0.8,
        max_scale=1.0)
    
    if options.debug:
        pass
        # print(json.dumps(result))
    box = (
        result['top_crop']['x'],
        result['top_crop']['y'],
        result['top_crop']['width'] + result['top_crop']['x'],
        result['top_crop']['height'] + result['top_crop']['y']
    )

    # Due to rounding issues, the box might be slightly to big
    # Scale it to fit the image correctly
    if image.size[0] < box[2]:
        print('Fixing Width: %s -> %s' % (box[2], image.size[0]))
        scale = image.size[0] / box[2]
        box = tuple(int(e * scale) for e in box)

    if image.size[1] < box[3]:
        print('Fixing Height: %s -> %s' % (box[3], image.size[1]))
        scale = image.size[1] / box[3]
        box = tuple(int(e * scale) for e in box)


    image = Image.open(options.inputfile)
    image2 = image.crop(box)
    image2.thumbnail((options.width, options.height), Image.ANTIALIAS)
    image2.save(options.outputfile, 'JPEG', quality=90)


if __name__ == '__main__':
    main()
