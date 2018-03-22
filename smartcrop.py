import argparse
import copy
import json
import math
import sys
from PIL import Image, ImageDraw
from PIL.ImageFilter import Kernel


CIE_TRANSFORM = (0.0722, 0.7152, 0.5126, 0)
LAPLACE_KERNEL = Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 0)


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
    'debug': False,
    'save_quality': 90,
    'file_type': 'JPEG'
}


def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = (x - 0.333333) * 8
    return max(1.0 - x * x, 0.0)


def saturation(r, g, b):
    maximum = max(r / 255., g / 255., b / 255.)
    minumum = min(r / 255., g / 255., b / 255.)
    if (maximum == minumum):
        return 0
    l = (maximum + minumum) / 2.
    d = maximum - minumum
    return d / (2 - maximum - minumum) if l > 0.5 else d / (maximum + minumum)


class SmartCrop(object):

    def __init__(self, options=DEFAULTS):
        self.options = options
        self.__thirds = [thirds(x / 500) for x in range(501)]

    def crop(self, image, options):
        if options['aspect']:
            options['width'] = options['aspect']
            options['height'] = 1

        scale = 1
        prescale = 1
        if options['width'] and options['height']:
            scale = min(image.size[0] / options['width'], image.size[1] / options['height'])
            options['crop_width'] = int(math.floor(options['width'] * scale))
            options['crop_height'] = int(math.floor(options['height'] * scale))
            # img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
            # don't set minscale smaller than 1/scale
            # -> don't pick crops that need upscaling
            options['min_scale'] = min(options['max_scale'] or SmartCrop.DEFAULTS.max_scale,
                                       max(1 / scale, (options['min_scale'] or SmartCrop.DEFAULTS.min_scale)))
        if options['width'] and options['height']:
            if options['prescale'] != False:
                prescale = 1. / scale / options['min_scale']
                if prescale < 1:
                    image.thumbnail(
                        (int(image.size[0] * prescale), int(image.size[1] * prescale)), Image.ANTIALIAS)
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
        mag = math.sqrt(r * r + g * g + b * b)
        options = self.options
        if mag == 0:
            rd = -options['skin_color'][0]
            gd = -options['skin_color'][1]
            bd = -options['skin_color'][2]
        else:
            rd = (r / mag - options['skin_color'][0])
            gd = (g / mag - options['skin_color'][1])
            bd = (b / mag - options['skin_color'][2])
        d = math.sqrt(rd * rd + gd * gd + bd * bd)
        return 1 - d

    def analyse(self, image):
        result = {}
        options = self.options

        bwimg = image.convert('L', CIE_TRANSFORM)

        edges = self.detect_edge(bwimg)
        skin = self.detect_skin(image, bwimg)
        saturation = self.detect_saturation(image, bwimg)
        _output = Image.merge('RGB', (skin,edges, saturation))

        if options['debug']:
            channels = _output.split()
            channels[0].save('channel0.jpg')
            channels[1].save('channel1.jpg')
            channels[2].save('channel2.jpg')

        score_output = copy.copy(_output)
        score_output.thumbnail((int(math.ceil(image.size[0] / options['score_down_sample'])),
                                int(math.ceil(image.size[1] / options['score_down_sample']))),
                               Image.ANTIALIAS)

        topScore = -sys.maxsize
        topCrop = None
        crops = self.crops(image)

        for crop in crops:
            crop['score'] = self.score(score_output, crop)
            if crop['score']['total'] > topScore:
                topCrop = crop
                topScore = crop['score']['total']

        result['crops'] = crops
        result['topCrop'] = topCrop

        if options['debug'] and topCrop:
            _debug_output = copy.copy(_output)
            _od = _debug_output.getdata()
            draw_image = Image.new("RGBA",
                                   (int(math.floor(topCrop['width'])),
                                    int(math.floor(topCrop['height']))), (255, 0, 0, 25))
            _d = ImageDraw.Draw(draw_image)
            # _d.rectangle(((topCrop['x'], topCrop['y']),
            _d.rectangle(((0, 0),
                          (topCrop['width'], topCrop['height'])),
                         outline=(255, 0, 0))
            for y in range(_output.size[1]):        # height
                for x in range(_output.size[0]):    # width
                    p = y * _output.size[0] + x
                    importance = self.importance(topCrop, x, y)
                    if importance > 0:
                        _od.putpixel(
                            (x, y), (_od[p][0], int(_od[p][1] + importance * 32), _od[p][2]))
                    if importance < 0:
                        _od.putpixel(
                            (x, y), (int(_od[p][0] + importance * -64), _od[p][1], _od[p][2]))
            _debug_output.paste(draw_image, (topCrop['x'], topCrop['y']), draw_image.split()[3])
            _debug_output.save('debug.jpg')
        return result

    def detect_edge(self, bwimg):
        return bwimg.filter(LAPLACE_KERNEL)

    def detect_skin(self, i, bw):
        o = Image.new('L', i.size)
        _id = i.getdata()
        _bwid = bw.getdata()
        w, h = i.size
        options = self.options
        for y in range(h):
            for x in range(w):
                p = y * w + x
                lightness = _bwid[p] / 255.
                skin = self.skin_color(_id[p][0], _id[p][1], _id[p][2])
                if skin > options['skin_threshold'] \
                        and lightness >= options['skin_brightness_min'] \
                        and lightness <= options['skin_brightness_max']:
                    o.putpixel((x, y), int((skin - options['skin_threshold']) * (255 / (1 - options['skin_threshold']))))
                else:
                    o.putpixel((x, y), 0)
        return o

    def detect_saturation(self, i, bw):
        o = Image.new('L', i.size)
        _id = i.getdata()
        _bwid = bw.getdata()
        w, h = i.size
        options = self.options
        for y in range(h):
            for x in range(w):
                p = y * w + x
                lightness = _bwid[p] / 255
                sat = saturation(_id[p][0], _id[p][1], _id[p][2])
                if sat > options['saturation_threshold'] \
                        and lightness >= options['saturation_brightness_min'] \
                        and lightness <= options['saturation_brightness_max']:
                    o.putpixel((x, y), int((sat - options['saturation_threshold']) * (255 / (1 - options['saturation_threshold']))))
                else:
                    o.putpixel((x, y), 0)
        return o

    def crops(self, image):
        crops = []
        width, height = image.size
        options = self.options
        minDimension = min(width, height)
        crop_width = options['crop_width'] or minDimension
        crop_height = options['crop_height'] or minDimension
        scales = [i / 100. for i in range(int(options['max_scale'] * 100),
                                          int((options['min_scale'] - options['scale_step']) * 100),
                                          -int(options['scale_step'] * 100))]
        for scale in scales:
            for y in range(0, height, options['step']):
                if not (y + crop_height * scale <= height):
                    break
                for x in range(0, width, options['step']):
                    if not (x + crop_width * scale <= width):
                        break
                    crops.append({
                        'x': x,
                        'y': y,
                        'width': crop_width * scale,
                        'height': crop_height * scale,
                    })
        return crops

    def score(self, output, crop):
        score = {'detail': 0,
                 'saturation': 0,
                 'skin': 0,
                 'total': 0,
                 }
        options = self.options
        od = output.getdata()
        downSample = options['score_down_sample']
        inv_downsample = 1. / downSample
        outputHeightDownSample = output.size[1] * downSample
        outputWidthDownSample = output.size[0] * downSample
        output_width = output.size[0]
        for y in range(0, outputHeightDownSample, downSample):
            for x in range(0, outputWidthDownSample, downSample):
                p = int(math.floor(y * inv_downsample) * output_width + math.floor(x * inv_downsample))
                importance = self.importance(crop, x, y)
                detail = od[p][1] / 255.
                score['skin'] += od[p][0] / 255. * (detail + options['skin_bias']) * importance
                score['detail'] += detail * importance
                score['saturation'] += od[p][2] / 255. * \
                    (detail + options['saturation_bias']) * importance
        score['total'] = (score['detail'] * options['detail_weight'] +
                          score['skin'] * options['skin_weight'] +
                          score['saturation'] * options['saturation_weight']) / crop['width'] / crop['height']
        return score

    def importance(self, crop, x, y):
        options = self.options

        if crop['x'] > x or x >= crop['x'] + crop['width'] or crop['y'] > y or y >= crop['y'] + crop['height']:
            return options['outside_importance']
        x = (x - crop['x']) / crop['width']
        y = (y - crop['y']) / crop['height']
        px = abs(0.5 - x) * 2
        py = abs(0.5 - y) * 2
        # distance from edge
        dx = max(px - 1.0 + options['edge_radius'], 0)
        dy = max(py - 1.0 + options['edge_radius'], 0)
        d = (dx * dx + dy * dy) * options['edge_weight']
        s = 1.41 - math.sqrt(px * px + py * py)
        if options['rule_of_thirds']:
            s += (max(0, s + d + 0.5) * 1.2) * (self.__thirds[int(px * 500)] + self.__thirds[int(py * 500)])
        return s + d


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='INPUT_FILE',
                        help='input image file')
    parser.add_argument('outputfile', metavar='OUTPUT_FILE',
                        help='output image file')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--width', dest='width', type=int, default=100, help='crop width')
    parser.add_argument('--height', dest='height', type=int, default=100, help='crop height')
    return parser.parse_args()


def main():
    opts = parse_argument()
    sc = SmartCrop()
    imgWidth = opts.width
    imgHeight = opts.height
    imgResizeFactor = imgWidth / 100.
    crop_options = DEFAULTS
    crop_options['debug'] = opts.debug
    crop_options['width'] = 100
    crop_options['height'] = int(imgHeight / imgResizeFactor)
    img = Image.open(opts.inputfile)
    if img.mode != 'RGB' and img.mode != 'RGBA':
        sys.stderr.write("{1} convert from mode='{0}' to mode='RGB'\n".format(img.mode, opts.inputfile))
        newimg = Image.new("RGB", img.size)
        newimg.paste(img)
        img = newimg
    ret = sc.crop(img, crop_options)
    if opts.debug:
        print(json.dumps(ret))
    box = (ret['topCrop']['x'],
           ret['topCrop']['y'],
           ret['topCrop']['width'] + ret['topCrop']['x'],
           ret['topCrop']['height'] + ret['topCrop']['y'])
    img = Image.open(opts.inputfile)
    img2 = img.crop(box)
    img2.thumbnail((imgWidth, imgHeight), Image.ANTIALIAS)
    img2.save(opts.outputfile, crop_options['file_type'], quality=crop_options['save_quality'])

if __name__ == '__main__':
    main()
