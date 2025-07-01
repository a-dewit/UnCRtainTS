import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
import rasterio
from rasterio.merge import merge
from s2cloudless import S2PixelCloudDetector

from data.apo_dataloader import CIRCA_from_HDF5
from data.process_functions import (
    S1_LAUNCH,
    get_cloud_map,
    iterdict,
    process_MS,
    process_SAR,
    read_img,
    read_tif,
    to_date,
)
from data.sampling_functions import (
    fixed_sampler,
    fixedsubset_sampler,
    random_sampler,
    sampler,
)

SEN12MSCRTS_SEQ_LENGTH = 30

""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN:
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    deprecated --> vary_samples:       bool, whether to draw random samples across epochs or not, matters only if sample_type is 'cloud_cloudfree'
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


class UnCRtainTS_from_hdf5(CIRCA_from_HDF5):
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

        assert self.__len__() > 0  # On s'assure que la donnée est bien trouvée
        assert sample_type in [
            "generic",
            "cloudy_cloudfree",
        ], "Input data must be either generic or cloudy_cloudfree type!"
        assert cloud_masks in [
            None,
            "cloud_cloudshadow_mask",
            "s2cloudless_map",
            "s2cloudless_mask",
        ], "Unknown cloud mask type!"
        self.modalities = ["S1", "S2"]
        self.time_points = range(SEN12MSCRTS_SEQ_LENGTH)
        self.cloud_masks = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type = sample_type if self.cloud_masks is not None else "generic"  # pick 'generic' or 'cloudy_cloudfree'
        self.sampling = sampler  # type of sampler
        self.vary_samples = self.sampling == "random" if self.sample_type == "cloudy_cloudfree" else False  # whether to draw different samples across epochs
        self.n_input_t = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input

        if self.vary_samples:
            self.t_windows = np.lib.stride_tricks.sliding_window_view(self.time_points, window_shape=self.n_input_t + 1)

        if self.cloud_masks in ["s2cloudless_map", "s2cloudless_mask"]:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else:
            self.cloud_detector = None

        self.method = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

    def incr_epoch_count(self):
        self.epoch_count += 1

    # load images at a given patch pdx for given time points tdx
    def get_imgs(self, pdx, tdx=range(0, 30)):
        # load the images and infer the masks
        s1_tif = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]["S1"])[tdx]]
        s2_tif = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]["S2"])[tdx]]
        coord = [list(tif.bounds) for tif in s2_tif]
        s1 = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2 = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection
        masks = None if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2]

        # get statistics and additional meta information
        coverage = [np.mean(mask) for mask in masks]
        s1_dates = [to_date(img.split("/")[-1].split("_")[5]) for img in np.array(self.paths[pdx]["S1"])[tdx]]
        s2_dates = [to_date(img.split("/")[-1].split("_")[5]) for img in np.array(self.paths[pdx]["S2"])[tdx]]
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
                in_s1_td = [(to_date(tdx[0].split("/")[-1].split("_")[-3]) - S1_LAUNCH).days for tdx in self.paths[pdx]["input"]["S1"]]
                in_s2_td = [(to_date(tdx[0].split("/")[-1].split("_")[-3]) - S1_LAUNCH).days for tdx in self.paths[pdx]["input"]["S2"]]
                tg_s1_td, tg_s2_td = [], []
                in_coord, tg_coord = [], []
                coverage_match = True

                custom = iterdict(self.custom_samples[pdx], self.mosaic_patches)

                input_s1 = np.array([process_SAR(img, self.method) for img in custom["input"]["S1"]])  # is of shape (T, C_S1, H, W)
                input_s2 = [process_MS(img, self.method) for img in custom["input"]["S2"]]  # is of shape (T, C_S2, H, W)
                input_masks = [] if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in custom["input"]["S2"]]

                target_s1 = process_SAR(custom["target"]["S1"], self.method)[0]
                target_s2 = [process_MS(custom["target"]["S2"], self.method)[0]]
                target_mask = [] if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in custom["input"]["S2"]]

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
                    coverage = [stats.item() for stats in self.data_pairs[adj_pdx]["coverage"]]

                    if self.sampling == "random":
                        inputs_idx, cloudless_idx, coverage_match = random_sampler(self.t_windows, coverage)
                    elif self.sampling == "fixedsubset":
                        inputs_idx, cloudless_idx, coverage_match = fixedsubset_sampler(self.n_input_t, self.min_cov, self.max_cov, coverage, earliest_idx=0, latext_idx=30)
                    else:  # default to fixed sampler
                        inputs_idx, cloudless_idx, coverage_match = fixed_sampler(self.n_input_t, self.min_cov, self.max_cov, coverage)

                    # if self.vary_samples: inputs_idx, cloudless_idx, coverage_match = self.random_sampler(coverage)
                    # else: inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(coverage)

                # INPUTS
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
                input_s1, input_s2, input_masks = np.array(in_s1), np.array(in_s2), np.array(in_masks)

                # TARGETS
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
                target_s1, target_s2, target_mask = np.array(tg_s1)[0], np.array(tg_s2)[0], np.array(tg_masks)[0]

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
                data_samples = sampler(self.sampling, self.t_windows, self.n_input_t, self.min_cov, self.max_cov, s1, s2, masks, coverage, clear_tresh=1e-3)

            if not self.custom_samples:
                input_s1, input_s2, input_masks, inputs_idx = data_samples["input"]
                target_s1, target_s2, target_mask, cloudless_idx = data_samples["target"]
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
                    "S1 path": ([] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]["S1"][idx]) for idx in inputs_idx]),
                    "S2 path": ([] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]["S2"][idx]) for idx in inputs_idx]),
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
                    "S1 path": ([] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]["S1"][cloudless_idx])]),
                    "S2 path": ([] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]["S2"][cloudless_idx])]),
                    "idx": [] if self.custom_samples else cloudless_idx,
                    "coord": tg_coord,  # [coord[cloudless_idx]],
                },
                "coverage bin": coverage_match,
            }

        elif self.sample_type == "generic":
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
                "S1 path": [os.path.join(self.root_dir, self.paths[pdx]["S1"][idx]) for idx in self.time_points],
                "S2 path": [os.path.join(self.root_dir, self.paths[pdx]["S2"][idx]) for idx in self.time_points],
                "coord": coord,
            }
        return sample
