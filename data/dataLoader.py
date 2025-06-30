import glob
import os
import warnings
from datetime import datetime
from typing import Optional, Union
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path

def to_date(string):
    return datetime.strptime(string, "%Y-%m-%d")


S1_LAUNCH = to_date("2014-04-03")

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
import rasterio
from rasterio.merge import merge
from s2cloudless import S2PixelCloudDetector
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


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


""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN:
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    depricated --> vary_samples:       bool, whether to draw random samples across epochs or not, matters only if sample_type is 'cloud_cloudfree'
    sampler             str, [fixed | fixedsubset | random]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    min_cov:            float, in [0.0, 1.0]
    max_cov:            float, in [0.0, 1.0]
    import_data_path:   str, path to importing the suppl. file specifying what time points to load for input and output

    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class UnCRtainTS_from_hdf5(Dataset):
    def __init__(
        self,
        phase: str = "all",
        hdf5_file: Optional[Union[str, Path]] = None,
        shuffle: bool = False,
        include_S1: bool = True,
        channels: Optional[str] = "all",

        cloud_masks="s2cloudless_mask",
        sample_type="cloudy_cloudfree",
        sampler="fixed",
        n_input_samples=3,
        rescale_method="default",
        min_cov=0.0,
        max_cov=1.0,
    ):

        super().__init__(
            phase=phase,
            hdf5_file=hdf5_file,
            shuffle=shuffle,
            include_S1=include_S1,
            channels=channels,
        )

        assert self.__len__() > 0 # On s'assure que la donnée est bien trouvée
        assert sample_type in ["generic", "cloudy_cloudfree"], (
            "Input data must be either generic or cloudy_cloudfree type!"
        )
        assert cloud_masks in [
            None,
            "cloud_cloudshadow_mask",
            "s2cloudless_map",
            "s2cloudless_mask",
        ], "Unknown cloud mask type!"

        self.modalities = ["S1", "S2"]
        self.time_points = range(30)
        self.cloud_masks = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type = (
            sample_type if self.cloud_masks is not None else "generic"
        )  # pick 'generic' or 'cloudy_cloudfree'
        self.sampling = sampler  # type of sampler
        self.vary_samples = (
            self.sampling == "random"
            if self.sample_type == "cloudy_cloudfree"
            else False
        )  # whether to draw different samples across epochs
        self.n_input_t = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input

        if self.vary_samples:
            self.t_windows = np.lib.stride_tricks.sliding_window_view(
                self.time_points, window_shape=self.n_input_t + 1
            )

        if self.cloud_masks in ["s2cloudless_map", "s2cloudless_mask"]:
            self.cloud_detector = S2PixelCloudDetector(
                threshold=0.4, all_bands=True, average_over=4, dilation_size=2
            )
        else:
            self.cloud_detector = None
        self.method = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

    def fixed_sampler(self, coverage, clear_tresh=1e-3):
        # sample custom time points from the current patch space in the current split
        # sort observation indices according to cloud coverage, ascendingly
        coverage_idx = np.argsort(coverage)
        cloudless_idx = coverage_idx[0]  # take the (earliest) least cloudy sample
        # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
        inputs_idx = [
            pdx
            for pdx, perc in enumerate(coverage)
            if perc >= self.min_cov and perc <= self.max_cov
        ][: self.n_input_t]
        if len(inputs_idx) < self.n_input_t:
            # ... if not exists then take the first n_input_t samples (except target patch)
            inputs_idx = [pdx for pdx in range(len(coverage)) if pdx != cloudless_idx][
                : self.n_input_t
            ]
            coverage_match = (
                False  # flag input samples that didn't meet the required cloud coverage
            )
        else:
            coverage_match = (
                True  # assume the requested amount of cloud coverage is met
            )
        # check whether the target meets the requested amount of clearness
        if coverage[cloudless_idx] > clear_tresh:
            coverage_match = False
        return inputs_idx, cloudless_idx, coverage_match

    def fixedsubset_sampler(
        self, coverage, earliest_idx=0, latext_idx=30, clear_tresh=1e-3
    ):
        # apply the fixed sampler on only a subsequence of the input sequence
        inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(
            self, coverage[earliest_idx:latext_idx], clear_tresh
        )
        # shift sampled indices by the offset of the subsequence
        inputs_idx, cloudless_idx = (
            [idx + earliest_idx for idx in inputs_idx],
            cloudless_idx + earliest_idx,
        )
        # if the sampled indices do not meet the criteria, then default to sampling over the full time series
        if not coverage_match:
            inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(
                self, coverage, clear_tresh
            )
        return inputs_idx, cloudless_idx, coverage_match

    def random_sampler(self, coverage, clear_tresh=1e-3):
        # sample a random target time point below 0.1% coverage (i.e. coverage<1e-3), or at min coverage
        is_clear = np.argwhere(np.array(coverage) < clear_tresh).flatten()
        try:
            cloudless_idx = is_clear[np.random.randint(0, len(is_clear))]
        except:
            cloudless_idx = np.array(coverage).argmin()
        # around this target time point, pick self.n_input_t input time points
        windows = [window for window in self.t_windows if cloudless_idx in window]
        # we pick the window with cloudless_idx centered such that input samples are temporally adjacent,
        # alternatively: pick a causal window (with cloudless_idx at the end) or randomly sample input dates
        inputs_idx = [
            input_t
            for input_t in windows[len(windows) // 2]
            if input_t != cloudless_idx
        ]
        coverage_match = True  # note: not checking whether any requested cloud coverage is met in this mode
        return inputs_idx, cloudless_idx, coverage_match

    def sampler(
        self, s1, s2, masks, coverage, clear_tresh=1e-3, earliest_idx=0, latext_idx=30
    ):
        if self.sampling == "random":
            inputs_idx, cloudless_idx, coverage_match = self.random_sampler(
                coverage, clear_tresh
            )
        elif self.sampling == "fixedsubset":
            inputs_idx, cloudless_idx, coverage_match = self.fixedsubset_sampler(
                coverage, clear_tresh, earliest_idx=earliest_idx, latext_idx=latext_idx
            )
        else:  # default to fixed sampler
            inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(
                coverage, clear_tresh
            )

        input_s1, input_s2, input_masks = (
            np.array(s1)[inputs_idx],
            np.array(s2)[inputs_idx],
            np.array(masks)[inputs_idx],
        )
        target_s1, target_s2, target_mask = (
            np.array(s1)[cloudless_idx],
            np.array(s2)[cloudless_idx],
            np.array(masks)[cloudless_idx],
        )

        data = {
            "input": [input_s1, input_s2, input_masks, inputs_idx],
            "target": [target_s1, target_s2, target_mask, cloudless_idx],
            "match": coverage_match,
        }
        return data

    # load images at a given patch pdx for given time points tdx
    def get_imgs(self, pdx, tdx=range(0, 30)):
        # load the images and infer the masks
        s1_tif = [
            read_tif(os.path.join(self.root_dir, img))
            for img in np.array(self.paths[pdx]["S1"])[tdx]
        ]
        s2_tif = [
            read_tif(os.path.join(self.root_dir, img))
            for img in np.array(self.paths[pdx]["S2"])[tdx]
        ]
        coord = [list(tif.bounds) for tif in s2_tif]
        s1 = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2 = [
            read_img(img) for img in s2_tif
        ]  # note: pre-processing happens after cloud detection
        masks = (
            None
            if not self.cloud_masks
            else [
                get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2
            ]
        )

        # get statistics and additional meta information
        coverage = [np.mean(mask) for mask in masks]
        s1_dates = [
            to_date(img.split("/")[-1].split("_")[5])
            for img in np.array(self.paths[pdx]["S1"])[tdx]
        ]
        s2_dates = [
            to_date(img.split("/")[-1].split("_")[5])
            for img in np.array(self.paths[pdx]["S2"])[tdx]
        ]
        s1_td = [(date - S1_LAUNCH).days for date in s1_dates]
        s2_td = [(date - S1_LAUNCH).days for date in s2_dates]

        return (
            s1_tif,
            s2_tif,
            coord,
            s1,
            s2,
            masks,
            coverage,
            s1_dates,
            s2_dates,
            s1_td,
            s2_td,
        )

    # function to merge (a temporal list of spatial lists containing) raster patches into a single rasterized patch
    def mosaic_patches(self, paths):
        src_files_to_mosaic = []

        for tp in paths:
            tp_mosaic = []
            for sp in tp:  # collect patches in space to mosaic over
                src = rasterio.open(os.path.join(self.root_dir, sp))
                tp_mosaic.append(src)
            mosaic, out_trans = merge(tp_mosaic)
            src_files_to_mosaic.append(mosaic.astype(np.float32))
        return src_files_to_mosaic  # , mosaic_meta

    def get_sample(self, pdx):
        """
            Retourne une donnée mais sans le pipeline de pre-processing
        """
        return self.__getitem__(pdx)

    def __getitem__(self, pdx):  # get the time series of one patch
        # get all images of patch pdx for online selection of dates tdx
        # s1_tif, s2_tif, coord, s1, s2, masks, coverage, s1_dates, s2_dates, s1_td, s2_td = self.get_imgs(pdx)
        
        patch_data = self.etl_item(item=pdx)
        # Select the correct channels
        if self.num_channels != patch_data["S2"]["S2"].shape[1]:
            patch_data["S2"]["S2"] = patch_data["S2"]["S2"][:, self.s2_channels, :, :]

        if self.sample_type == "cloudy_cloudfree":
            # this sample type allows for four manners of sampling data:
            # a) by loading custom-defined samples, b.i) & b.ii) based on importing pre-computed statistics, and c) for full online computations
            if self.custom_samples:
                in_s1_td = [
                    (to_date(tdx[0].split("/")[-1].split("_")[-3]) - S1_LAUNCH).days
                    for tdx in self.paths[pdx]["input"]["S1"]
                ]
                in_s2_td = [
                    (to_date(tdx[0].split("/")[-1].split("_")[-3]) - S1_LAUNCH).days
                    for tdx in self.paths[pdx]["input"]["S2"]
                ]
                tg_s1_td, tg_s2_td = [], []
                in_coord, tg_coord = [], []
                coverage_match = True

                custom = iterdict(self.custom_samples[pdx], self.mosaic_patches)

                input_s1 = np.array(
                    [process_SAR(img, self.method) for img in custom["input"]["S1"]]
                )  # is of shape (T, C_S1, H, W)
                input_s2 = [
                    process_MS(img, self.method) for img in custom["input"]["S2"]
                ]  # is of shape (T, C_S2, H, W)
                input_masks = (
                    []
                    if not self.cloud_masks
                    else [
                        get_cloud_map(img, self.cloud_masks, self.cloud_detector)
                        for img in custom["input"]["S2"]
                    ]
                )

                target_s1 = process_SAR(custom["target"]["S1"], self.method)[0]
                target_s2 = [process_MS(custom["target"]["S2"], self.method)[0]]
                target_mask = (
                    []
                    if not self.cloud_masks
                    else [
                        get_cloud_map(img, self.cloud_masks, self.cloud_detector)
                        for img in custom["input"]["S2"]
                    ]
                )

            elif self.import_data_path:
                # compute epoch-sensitive index, wrap-around if exceeds imported dates
                adj_pdx = (self.epoch_count * self.__len__() + pdx) % self.n_data_pairs

                if "input" in self.data_pairs and "target" in self.data_pairs:
                    # b.i) import pre-computed date indices:
                    #   1. read pre-computed date indices
                    #   2. only read images and compute masks of pre-computed dates tdx for patch pdx
                    inputs_idx, cloudless_idx, coverage_match = (
                        self.data_pairs[adj_pdx]["input"],
                        self.data_pairs[adj_pdx]["target"],
                        True,
                    )
                else:
                    # b.ii) import pre-computed cloud coverage:
                    #   1. read pre-computed cloud coverage
                    #   2. sample dates tdx online, given cloud coverage
                    #   3. only read images and compute masks of pre-computed dates tdx for patch pdx
                    coverage = [
                        stats.item() for stats in self.data_pairs[adj_pdx]["coverage"]
                    ]

                    if self.sampling == "random":
                        inputs_idx, cloudless_idx, coverage_match = self.random_sampler(
                            coverage
                        )
                    elif self.sampling == "fixedsubset":
                        inputs_idx, cloudless_idx, coverage_match = (
                            self.fixedsubset_sampler(
                                coverage, earliest_idx=0, latext_idx=30
                            )
                        )
                    else:  # default to fixed sampler
                        inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(
                            coverage
                        )

                    # if self.vary_samples: inputs_idx, cloudless_idx, coverage_match = self.random_sampler(coverage)
                    # else: inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(coverage)

                (
                    in_s1_tif,
                    in_s2_tif,
                    in_coord,
                    in_s1,
                    in_s2,
                    in_masks,
                    in_coverage,
                    in_s1_dates,
                    in_s2_dates,
                    in_s1_td,
                    in_s2_td,
                ) = self.get_imgs(pdx, inputs_idx)
                (
                    tg_s1_tif,
                    tg_s2_tif,
                    tg_coord,
                    tg_s1,
                    tg_s2,
                    tg_masks,
                    tg_coverage,
                    tg_s1_dates,
                    tg_s2_dates,
                    tg_s1_td,
                    tg_s2_td,
                ) = self.get_imgs(pdx, [cloudless_idx])

                target_s1, target_s2, target_mask = (
                    np.array(tg_s1)[0],
                    np.array(tg_s2)[0],
                    np.array(tg_masks)[0],
                )
                input_s1, input_s2, input_masks = (
                    np.array(in_s1),
                    np.array(in_s2),
                    np.array(in_masks),
                )

                data_samples = {
                    "input": [input_s1, input_s2, input_masks, inputs_idx],
                    "target": [target_s1, target_s2, target_mask, cloudless_idx],
                    "match": coverage_match,
                }
            else:
                # c) infer date indices online:
                #   1. read all images and compute every mask indiscriminately
                #   2. post-hoc select the most optimal dates tdx for patch pdx
                (
                    s1_tif,
                    s2_tif,
                    coord,
                    s1,
                    s2,
                    masks,
                    coverage,
                    s1_dates,
                    s2_dates,
                    s1_td,
                    s2_td,
                ) = self.get_imgs(pdx)
                data_samples = self.sampler(s1, s2, masks, coverage, clear_tresh=1e-3)

            if not self.custom_samples:
                input_s1, input_s2, input_masks, inputs_idx = data_samples["input"]
                target_s1, target_s2, target_mask, cloudless_idx = data_samples[
                    "target"
                ]
                coverage_match = data_samples["match"]

                # preprocess S2 data (after cloud masks have been computed)
                input_s2 = [process_MS(img, self.method) for img in input_s2]
                target_s2 = [process_MS(target_s2, self.method)]

                if not self.import_data_path:
                    in_s1_td, in_s2_td = (
                        [s1_td[idx] for idx in inputs_idx],
                        [s2_td[idx] for idx in inputs_idx],
                    )
                    tg_s1_td, tg_s2_td = [s1_td[cloudless_idx]], [s2_td[cloudless_idx]]
                    in_coord, tg_coord = (
                        [coord[idx] for idx in inputs_idx],
                        [coord[cloudless_idx]],
                    )

            sample = {
                "input": {
                    "S1": list(input_s1),
                    "S2": input_s2,
                    "masks": list(input_masks),
                    "coverage": [np.mean(mask) for mask in input_masks],
                    "S1 TD": in_s1_td,  # [s1_td[idx] for idx in inputs_idx],
                    "S2 TD": in_s2_td,  # [s2_td[idx] for idx in inputs_idx],
                    "S1 path": []
                    if self.custom_samples
                    else [
                        os.path.join(self.root_dir, self.paths[pdx]["S1"][idx])
                        for idx in inputs_idx
                    ],
                    "S2 path": []
                    if self.custom_samples
                    else [
                        os.path.join(self.root_dir, self.paths[pdx]["S2"][idx])
                        for idx in inputs_idx
                    ],
                    "idx": [] if self.custom_samples else inputs_idx,
                    "coord": in_coord,  # [coord[idx] for idx in inputs_idx],
                },
                "target": {
                    "S1": [target_s1],
                    "S2": target_s2,
                    "masks": [target_mask],
                    "coverage": [np.mean(target_mask)],
                    "S1 TD": tg_s1_td,  # [s1_td[cloudless_idx]],
                    "S2 TD": tg_s2_td,  # [s2_td[cloudless_idx]],
                    "S1 path": []
                    if self.custom_samples
                    else [
                        os.path.join(
                            self.root_dir, self.paths[pdx]["S1"][cloudless_idx]
                        )
                    ],
                    "S2 path": []
                    if self.custom_samples
                    else [
                        os.path.join(
                            self.root_dir, self.paths[pdx]["S2"][cloudless_idx]
                        )
                    ],
                    "idx": [] if self.custom_samples else cloudless_idx,
                    "coord": tg_coord,  # [coord[cloudless_idx]],
                },
                "coverage bin": coverage_match,
            }

        elif self.sample_type == "generic":
            # did not implement custom sampling for options other than 'cloudy_cloudfree' yet
            if self.custom_samples:
                raise NotImplementedError

            (
                s1_tif,
                s2_tif,
                coord,
                s1,
                s2,
                masks,
                coverage,
                s1_dates,
                s2_dates,
                s1_td,
                s2_td,
            ) = self.get_imgs(pdx)

            sample = {
                "S1": s1,
                "S2": [process_MS(img, self.method) for img in s2],
                "masks": masks,
                "coverage": coverage,
                "S1 TD": s1_td,
                "S2 TD": s2_td,
                "S1 path": [
                    os.path.join(self.root_dir, self.paths[pdx]["S1"][idx])
                    for idx in self.time_points
                ],
                "S2 path": [
                    os.path.join(self.root_dir, self.paths[pdx]["S2"][idx])
                    for idx in self.time_points
                ],
                "coord": coord,
            }
        return sample

    def __len__(self):
        # length of generated list
        return len(self.self.patches_dataset)

    def incr_epoch_count(self):
        # increment epoch count by 1
        self.epoch_count += 1