import warnings
from datetime import datetime
import numpy as np

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
import rasterio
from scipy.ndimage import gaussian_filter
from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


def to_date(string):
    return datetime.strptime(string, "%Y-%m-%d")


S1_LAUNCH = to_date("2014-04-03")


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif


def read_img(tif):
    return tif.read().astype(np.float32)


def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img = (img - oldMin) / oldRange
    return img


def process_MS(img, method):
    if method == "default":
        intensity_min, intensity_max = (
            0,
            10000,
        )  # define a reasonable range of MS intensities
        img = np.clip(
            img, intensity_min, intensity_max
        )  # intensity clipping to a global unified MS intensity range
        img = rescale(
            img, intensity_min, intensity_max
        )  # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method == "resnet":
        intensity_min, intensity_max = (
            0,
            10000,
        )  # define a reasonable range of MS intensities
        img = np.clip(
            img, intensity_min, intensity_max
        )  # intensity clipping to a global unified MS intensity range
        img /= 2000  # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return img


def process_SAR(img, method):
    if method == "default":
        dB_min, dB_max = -25, 0  # define a reasonable range of SAR dB
        img = np.clip(
            img, dB_min, dB_max
        )  # intensity clipping to a global unified SAR dB range
        img = rescale(
            img, dB_min, dB_max
        )  # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method == "resnet":
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate(
            [
                (
                    2
                    * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0])
                    / (dB_max[0] - dB_min[0])
                )[None, ...],
                (
                    2
                    * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1])
                    / (dB_max[1] - dB_min[1])
                )[None, ...],
            ],
            axis=0,
        )
    img = np.nan_to_num(img)
    return img


def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask


# recursively apply function to nested dictionary
def iterdict(dictionary, fct):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dictionary[k] = iterdict(v, fct)
        else:
            dictionary[k] = fct(v)
    return dictionary


def get_cloud_map(img, detector, instance=None):
    # get cloud masks
    img = np.clip(img, 0, 10000)
    mask = np.ones((img.shape[-1], img.shape[-1]))
    # note: if your model may suffer from dark pixel artifacts,
    #       you may consider adjusting these filtering parameters
    if not (img.mean() < 1e-5 and img.std() < 1e-5):
        if detector == "cloud_cloudshadow_mask":
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = get_cloud_cloudshadow_mask(img, threshold)
        elif detector == "s2cloudless_map":
            threshold = 0.5
            mask = instance.get_cloud_probability_maps(
                np.moveaxis(img / 10000, 0, -1)[None, ...]
            )[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2)
        elif detector == "s2cloudless_mask":
            mask = instance.get_cloud_masks(np.moveaxis(img / 10000, 0, -1)[None, ...])[
                0, ...
            ]
        else:
            mask = np.ones((img.shape[-1], img.shape[-1]))
            warnings.warn(f"Method {detector} not yet implemented!")
    else:
        warnings.warn("Encountered a blank sample, defaulting to cloudy mask.")
    return mask.astype(np.float32)