from __future__ import annotations
from dataclasses import dataclass
from functools import cache
import math
import sys

import numpy as np
from PIL import Image
from PIL.ImageFilter import Kernel


DEFAULT_SKIN_COLOR = (0.78, 0.57, 0.44)


def saturation(image) -> np.ndarray:
    r, g, b = image.split()
    r, g, b = np.array(r), np.array(g), np.array(b)
    r, g, b = r.astype(float), g.astype(float), b.astype(float)
    maximum = np.maximum(np.maximum(r, g), b)  # [0; 255]
    minimum = np.minimum(np.minimum(r, g), b)  # [0; 255]
    s = (maximum + minimum) / 255  # [0.0; 1.0] pylint:disable=invalid-name
    d = (maximum - minimum) / 255  # [0.0; 1.0] pylint:disable=invalid-name
    s[maximum == minimum] = 0.001  # avoid division by zero
    mask = s > 1
    s[mask] = 2 - s[mask]
    return d / s  # [0.0; 1.0]

@cache
def thirds(x) -> float:
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = 8 * (x + 2 / 3) - 8    # 8*x-8/3 is even simpler, but with ~e-16 floating error
    return max(1 - x * x, 0)

# a quite odd workaround for using slots for python > 3.9
@dataclass(eq=False, **{"slots": True} if sys.version_info.minor > 9 else {})
class SmartCrop:  # pylint:disable=too-many-instance-attributes
    detail_weight: float = 0.2
    edge_radius: float = 0.4
    edge_weight: float = -20
    outside_importance: float = -0.5
    rule_of_thirds: bool = True
    saturation_bias: float = 0.2
    saturation_brightness_max: float = 0.9
    saturation_brightness_min: float = 0.05
    saturation_threshold: float = 0.4
    saturation_weight: float = 0.3
    score_down_sample: int = 8
    skin_bias: float = 0.01
    skin_brightness_max: float = 1
    skin_brightness_min: float = 0.2
    skin_color: tuple[float, float, float] = DEFAULT_SKIN_COLOR
    skin_threshold: float = 0.8
    skin_weight: float = 1.8

    def analyse(  # pylint:disable=too-many-arguments,too-many-locals
        self,
        image,
        crop_width: int,
        crop_height: int,
        *,
        max_scale: float = 1,
        min_scale: float = 0.9,
        scale_step: float = 0.1,
        step: int = 8
    ) -> dict:
        """
        Analyze image and return some suggestions of crops (coordinates).
        This implementation / algorithm is really slow for large images.
        Use `crop()` which is pre-scaling the image before analyzing it.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        analyse_image = self.prepare_features_image(image)
        score_image = analyse_image.resize(
            (
                int(math.ceil(image.size[0] / self.score_down_sample)),
                int(math.ceil(image.size[1] / self.score_down_sample))
            ),
            Image.Resampling.LANCZOS)

        crops = self.crops(
            image,
            crop_width,
            crop_height,
            max_scale=max_scale,
            min_scale=min_scale,
            scale_step=scale_step,
            step=step)

        for crop in crops:
            crop['score'] = self.score(score_image, crop)

        top_crop = max(crops, key=lambda c: c['score']['total'])

        return {'analyse_image': analyse_image, 'crops': crops, 'top_crop': top_crop}

    def crop(  # pylint:disable=too-many-arguments,too-many-locals
        self,
        image,
        width: int,
        height: int,
        *,
        prescale: bool = True,
        max_scale: float = 1,
        min_scale: float = 0.9,
        scale_step: float = 0.1,
        step: int = 8
    ) -> dict:
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
                image = image.resize(
                    (int(image.size[0] * prescale_size), int(image.size[1] * prescale_size)),
                    Image.Resampling.LANCZOS)
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

    def crops(  # pylint:disable=too-many-arguments
        self,
        image,
        crop_width: int,
        crop_height: int,
        *,
        max_scale: float = 1,
        min_scale: float = 0.9,
        scale_step: float = 0.1,
        step: int = 8
    ) -> list[dict]:
        image_width, image_height = image.size
        crops = []
        for scale in (
            i / 100 for i in range(
                int(max_scale * 100),
                int((min_scale - scale_step) * 100),
                -int(scale_step * 100))
        ):
            for y in range(0, image_height, step):
                if y + crop_height * scale > image_height:
                    break
                for x in range(0, image_width, step):
                    if x + crop_width * scale > image_width:
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

    def debug_crop(self, analyse_image, crop: dict, orig_size: tuple[int, int]) -> Image:
        debug_image = analyse_image.copy()
        debug_pixels = debug_image.getdata()

        ratio_horizontal = debug_image.size[0] / orig_size[0]
        ratio_vertical = debug_image.size[1] / orig_size[1]
        fake_crop = {
            'x': crop['x'] * ratio_horizontal,
            'y': crop['y'] * ratio_vertical,
            'width': crop['width'] * ratio_horizontal,
            'height': crop['height'] * ratio_vertical,
        }

        for y in range(analyse_image.size[1]):        # height
            for x in range(analyse_image.size[0]):    # width
                index = y * analyse_image.size[0] + x
                importance = self.importance(fake_crop, x, y)
                redder, greener = (-64, 0) if importance < 0 else (0, 32)
                debug_pixels.putpixel(
                    (x, y),
                    (
                        debug_pixels[index][0] + int(importance * redder),
                        debug_pixels[index][1] + int(importance * greener),
                        debug_pixels[index][2]
                    ))

        # in case you want a whitish outline to mark the crop
        # ImageDraw.Draw(debug_image).rectangle([fake_crop['x'],
        #                                        fake_crop['y'],
        #                                        fake_crop['x'] + fake_crop['width'],
        #                                        fake_crop['y'] + fake_crop['height']],
        #                                        outline=(175, 175, 175), width=2)

        return debug_image

    def prepare_features_image(self, image: Image) -> Image:
        # luminance
        cie_image = image.convert('L', (0.2126, 0.7152, 0.0722, 0))
        cie_array = np.asarray(cie_image)  # [0; 255]

        return Image.merge(
            mode='RGB',
            bands=(
                self.detect_skin(cie_array, image),
                self.detect_edge(cie_image),
                self.detect_saturation(cie_array, image),
            )
        )

    def detect_edge(self, cie_image) -> Image:
        return cie_image.filter(Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 1))

    def detect_saturation(self, cie_array: np.ndarray, source_image) -> Image:
        threshold = self.saturation_threshold
        saturation_data = saturation(source_image)
        mask = (
            (saturation_data > threshold) &
            (cie_array >= self.saturation_brightness_min * 255) &
            (cie_array <= self.saturation_brightness_max * 255))

        saturation_data[~mask] = 0
        saturation_data[mask] = (saturation_data[mask] - threshold) * (255 / (1 - threshold))

        return Image.fromarray(saturation_data.astype('uint8'))

    def detect_skin(self, cie_array: np.ndarray, source_image) -> Image:
        r, g, b = source_image.split()
        r, g, b = np.array(r), np.array(g), np.array(b)
        r, g, b = r.astype(float), g.astype(float), b.astype(float)

        mag = np.sqrt(r * r + g * g + b * b) + 0.001   # avoid division by zero
        rd = r / mag - self.skin_color[0]
        gd = g / mag - self.skin_color[1]
        bd = b / mag - self.skin_color[2]

        skin = 1 - np.sqrt(rd * rd + gd * gd + bd * bd)
        mask = (
            (skin > self.skin_threshold) &
            (cie_array >= self.skin_brightness_min * 255) &
            (cie_array <= self.skin_brightness_max * 255))

        skin_data = (skin - self.skin_threshold) * (255 / (1 - self.skin_threshold))
        skin_data[~mask] = 0

        return Image.fromarray(skin_data.astype('uint8'))

    def importance(self, crop: dict, x: int, y: int) -> float:
        if (
            crop['x'] > x or x >= crop['x'] + crop['width'] or
            crop['y'] > y or y >= crop['y'] + crop['height']
        ):
            return self.outside_importance

        x = (x - crop['x']) / crop['width']
        y = (y - crop['y']) / crop['height']
        px, py = abs(0.5 - x) * 2, abs(0.5 - y) * 2  # pylint:disable=invalid-name

        # distance from edge
        dx = max(px - 1 + self.edge_radius, 0)      # pylint:disable=invalid-name
        dy = max(py - 1 + self.edge_radius, 0)      # pylint:disable=invalid-name
        d = (dx * dx + dy * dy) * self.edge_weight  # pylint:disable=invalid-name
        s = 1.41 - math.sqrt(px * px + py * py)     # pylint:disable=invalid-name

        if self.rule_of_thirds:
            # pylint:disable=invalid-name
            s += (max(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py))

        return s + d

    def score(self, target_image, crop: dict) -> dict:  # pylint:disable=too-many-locals
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
                index = int(
                    math.floor(y * inv_down_sample) * target_width +
                    math.floor(x * inv_down_sample)
                )
                importance = self.importance(crop, x, y)
                detail = target_data[index][1] / 255
                score['skin'] += (
                    target_data[index][0] / 255 * (detail + self.skin_bias) * importance
                )
                score['detail'] += detail * importance
                score['saturation'] += (
                    target_data[index][2] / 255 * (detail + self.saturation_bias) * importance
                )
        score['total'] = (
            score['detail'] * self.detail_weight +
            score['skin'] * self.skin_weight +
            score['saturation'] * self.saturation_weight
        ) / (crop['width'] * crop['height'])

        return score
