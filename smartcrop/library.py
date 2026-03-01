from __future__ import annotations
from dataclasses import dataclass
import math
import sys

import numpy as np
from PIL import Image
from PIL.ImageFilter import Kernel


DEFAULT_SKIN_COLOR = (0.78, 0.57, 0.44)


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
        num_scale_steps: int = 2,
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
        downsampled_features = analyse_image.resize(
            (
                int(math.ceil(image.size[0] / self.score_down_sample)),
                int(math.ceil(image.size[1] / self.score_down_sample))
            ),
            Image.Resampling.LANCZOS)

        precomputed_features = self.precompute_features(downsampled_features)
        features_sum = np.sum(precomputed_features)
        prescore = features_sum * self.outside_importance

        crops = self.crops(
            image,
            crop_width,
            crop_height,
            max_scale=max_scale,
            min_scale=min_scale,
            num_scale_steps=num_scale_steps,
            step=step
        )

        cached_importances = {}
        inv_down_sample = 1 / self.score_down_sample

        for crop in crops:
            cx, cy, cw, ch = map(
                lambda val: int(val * inv_down_sample),
                [crop['x'], crop['y'], crop['width'], crop['height']]
            )

            if (cw, ch) not in cached_importances:
                cached_importances[(cw, ch)] = self.get_importance(
                    width=cw, height=ch
                ) - self.outside_importance
            importance = cached_importances[(cw, ch)]

            crop['score'] = self.score(
                precomputed_features, prescore, (cx, cy, cw, ch), importance
            )

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
        num_scale_steps: int = 2,
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
            num_scale_steps=num_scale_steps,
            step=step)

        for i in range(len(result['crops'])):
            crop = result['crops'][i]
            crop['x'] = int(math.floor(crop['x'] / prescale_size))
            crop['y'] = int(math.floor(crop['y'] / prescale_size))
            crop['width'] = int(math.floor(crop['width'] / prescale_size))
            crop['height'] = int(math.floor(crop['height'] / prescale_size))
            result['crops'][i] = crop
        return result

    def crops(  # pylint:disable=too-many-arguments,too-many-locals
        self,
        image,
        crop_width: int,
        crop_height: int,
        *,
        max_scale: float = 1,
        min_scale: float = 0.9,
        num_scale_steps: int = 2,
        step: int = 8
    ) -> list[dict]:
        """
        Generate a list of potential crop coordinates across different scales
        for a given image size. Please note that the following conditions must
        be met: 0 < min_scale ≤ max_scale ≤ 1, num_scale_steps > 0
        """
        if not isinstance(num_scale_steps, int):
            raise ValueError(
                f'num_scale_steps should be an integer! Got: {type(num_scale_steps).__name__}'
            )
        if not isinstance(step, int):
            raise ValueError(
                f'step should be an integer! Got: {type(step).__name__}'
            )
        if num_scale_steps < 1:
            raise ValueError(
                f'num_scale_steps must be at least 1! Got: {num_scale_steps}'
            )
        if max_scale == min_scale and num_scale_steps > 1:
            num_scale_steps = 1
        if not 0 < min_scale <= max_scale <= 1:
            op1 = '!' if max_scale > 1 else '≤'
            op2 = '!' if min_scale > max_scale else '≤'
            op3 = '!' if min_scale <= 0 else '<'

            def f(num):
                s = str(num).rjust(9)[:9]
                return s if s[0] == ' ' else s[:8] + '…'

            raise ValueError(
                'Bad scale bounds!\n'
                f'  Expected: 0 < min_scale ≤ max_scale ≤ 1\n'
                f'  Received: 0 {op3} {f(min_scale)} {op2} {f(max_scale)} {op1} 1'
            )

        image_width, image_height = image.size
        crops = []
        last_crop_size = None
        for scale in np.linspace(max_scale, min_scale, num_scale_steps):
            crop_size = (
                math.ceil(crop_width * scale), math.ceil(crop_height * scale)
            )
            if last_crop_size == crop_size:
                continue
            last_crop_size = crop_size
            for y in range(0, image_height - crop_size[1] + 1, step):
                for x in range(0, image_width - crop_size[0] + 1, step):
                    crops.append({
                        'x': x,
                        'y': y,
                        'width': crop_size[0],
                        'height': crop_size[1],
                    })
        if not crops:
            raise ValueError(locals())
        return crops

    def debug_crop(self, analyse_image, crop: dict, orig_size: tuple[int, int]) -> Image:
        """
        Creates a debug visualization showing how importance weights affect a
        specific crop region. This function is intended to be used for internal
        debugging. The original image dimensions `orig_size` are required to
        correctly prescale the crop coordinates.
        """
        ratio_horizontal = analyse_image.size[0] / orig_size[0]
        ratio_vertical = analyse_image.size[1] / orig_size[1]
        i_x, i_width, = map(
            lambda n: int(n * ratio_horizontal), (crop['x'], crop['width'])
        )
        i_y, i_height = map(
            lambda n: int(n * ratio_vertical), (crop['y'], crop['height'])
        )

        features_data = np.array(analyse_image, dtype=np.float32)
        importance_map = self.get_importance(height=i_height, width=i_width)

        # window there the importance is applied
        i_window = features_data[i_y : i_y + i_height, i_x : i_x + i_width]  # noqa: E203

        # place the outside importance
        features_data += np.array([-64 * self.outside_importance, 0, 0])

        # apply the importance on the window
        mask = importance_map > 0
        i_window[~mask, 0] += -64 * importance_map[~mask]   # redder
        i_window[mask, 1] += 32 * importance_map[mask]      # greener
        features_data[i_y : i_y + i_height, i_x : i_x + i_width] = i_window  # noqa: E203

        return Image.fromarray(np.clip(features_data, 0, 255).astype(np.uint8))

    def prepare_features_image(self, image: Image) -> Image:
        # luminance
        cie_image = image.convert('L', (0.2126, 0.7152, 0.0722, 0))
        cie_array = np.asarray(cie_image, dtype=np.float32)  # [0; 255]
        image_array = np.array(image, dtype=np.float32)

        return Image.merge(
            mode='RGB',
            bands=(
                self.detect_skin(cie_array, image_array),
                self.detect_edge(cie_image),
                self.detect_saturation(cie_array, image_array),
            )
        )

    @staticmethod
    def detect_feature(
            feature_data: np.ndarray,
            threshold: float,
            min_cie: float,
            max_cie: float,
            cie_array: np.ndarray
    ) -> np.ndarray:
        mask = (
            (feature_data > threshold) &
            (cie_array >= min_cie * 255) &
            (cie_array <= max_cie * 255)
        )
        feature_data = (feature_data - threshold) * (255 / (1 - threshold))
        feature_data[~mask] = 0

        return Image.fromarray(feature_data.astype(np.uint8))

    def detect_edge(self, cie_image) -> Image:
        return cie_image.filter(Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 1))

    def detect_saturation(self, cie_array: np.ndarray, source_image: np.ndarray) -> Image:
        r = source_image[..., 0]
        g = source_image[..., 1]
        b = source_image[..., 2]

        maximum = np.maximum(np.maximum(r, g), b)  # [0; 255]
        minimum = np.minimum(np.minimum(r, g), b)  # [0; 255]
        s = (maximum + minimum) / 255  # [0.0; 1.0] pylint:disable=invalid-name
        d = (maximum - minimum) / 255  # [0.0; 1.0] pylint:disable=invalid-name
        s[maximum == minimum] = 0.001  # avoid division by zero
        mask = s > 1
        s[mask] = 2 - s[mask]
        saturation_data = d / s  # [0.0; 1.0]

        return SmartCrop.detect_feature(
            feature_data=saturation_data,
            threshold=self.saturation_threshold,
            min_cie=self.saturation_brightness_min,
            max_cie=self.saturation_brightness_max,
            cie_array=cie_array,
        )

    def detect_skin(self, cie_array: np.ndarray, source_image: np.ndarray) -> Image:
        r = source_image[..., 0]
        g = source_image[..., 1]
        b = source_image[..., 2]

        mag = np.sqrt(r * r + g * g + b * b) + 0.001   # avoid division by zero
        rd = r / mag - self.skin_color[0]
        gd = g / mag - self.skin_color[1]
        bd = b / mag - self.skin_color[2]

        skin_data = 1 - np.sqrt(rd * rd + gd * gd + bd * bd)

        return SmartCrop.detect_feature(
            feature_data=skin_data,
            threshold=self.skin_threshold,
            min_cie=self.skin_brightness_min,
            max_cie=self.skin_brightness_max,
            cie_array=cie_array,
        )

    def get_importance(self, height, width) -> np.ndarray:
        """
        Generate composite weighting map for a scoring crop.
        """
        # the original importance has a scaling that not include 1.0
        xx = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)
        px = np.abs(0.5 - xx) * 2
        py = np.abs(0.5 - yy) * 2
        edge_threshold = 1.0 - self.edge_radius
        dx = np.maximum(px - edge_threshold, 0.0)
        dy = np.maximum(py - edge_threshold, 0.0)
        d = (np.square(dy[:, np.newaxis]) + np.square(dx)) * self.edge_weight
        # 1.41 is just an approximation of the square root of 2, no magic
        s = 1.41 - np.sqrt(np.square(py[:, np.newaxis]) + np.square(px))

        if self.rule_of_thirds:
            def thirds(t):
                # that's kind of parabola centered at 1/3
                t = 1.0 - 64.0 * np.square(t - 1.0 / 3)
                return np.maximum(t, 0.0)
            # 1.2 is pure magic from original js code
            thirds_weight = (thirds(py)[:, np.newaxis] + thirds(px)) * 1.2
            intermediate = s + d + 0.5
            s += np.maximum(intermediate, 0.0) * thirds_weight

        return s + d

    def precompute_features(self, features_image: Image) -> np.ndarray:
        """
        Apply scaling, biasing, and weighting transformations to image features.
        """
        features = np.array(features_image).astype(np.float32)
        inv255 = 1 / 255
        features *= inv255

        skin = features[..., 0]
        detail = features[..., 1]
        satur = features[..., 2]

        skin *= detail + self.skin_bias
        satur *= detail + self.saturation_bias

        precomputed = (
            skin * self.skin_weight +
            detail * self.detail_weight +
            satur * self.saturation_weight
        )

        return precomputed

    def score(
        self,
        features_data: np.ndarray,
        prescore: np.ndarray,
        crop_dimensions: tuple[int, int, int, int],  # (x, y, w, h)
        importance: np.ndarray
    ) -> dict:  # pylint:disable=too-many-locals
        """
        Calculates a score for a crop region and returns it in a dictionary.
        """
        x, y, w, h = crop_dimensions

        score = prescore + np.sum(
            features_data[y: y + h, x: x + w] * importance
        )
        total = score / (w * h)

        return {'total': total}
