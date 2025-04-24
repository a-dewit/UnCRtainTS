import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
import dataloader_CIRCA.tools.positional_encoding as encodings 
import dataloader_CIRCA.tools.torch_transforms as torch_transforms
from dataloader_CIRCA.datasets import CircaPatchDataSet

CHANNEL_CONFIG = ['bgr', 'bgr-nir', 'all', 'bgr-mask', 'bgr-nir-mask', 'all-mask']


class UTILISEDataset(CircaPatchDataSet):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks. Dataset adapted for U-TILISE repo.
    """

    def __init__(
        self,
        data_optique: Union[str, Path],
        data_radar: Union[str, Path],
        patch_size: int = 256,
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_SAR: bool = False,
        channels: str = "all",
        augment: bool = False,
        seq_length: int = 30,
        pe_strategy: str = 'day-of-year',
    ):
        """
        Initializes the dataset.

        Parameters:
        - data_optique (Union[str, Path]): Path to the optical data directory.
        - data_radar (Union[str, Path]): Path to the radar data directory.
        - patch_size (int, optional): Size of the patches to extract. Defaults to 256.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        - use_SAR (bool, optional): Whether to include SAR data in the dataset. Defaults to True.
        - channels(str): Channels to be extracted, choose among ['rgb', 'bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask'].

        """
        super(UTILISEDataset, self).__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            patch_size=patch_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_SAR=use_SAR,
        )
        # Init des variables propres à U-TILISE
        self.seq_length = seq_length
        self.pe_strategy = pe_strategy
        self.set_channels(channels)
        self.set_augment(augment)


    def set_augment(self, augment):
        if augment:
            self.augmentation_function = transforms.Compose([
                torch_transforms.Rotate(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.augmentation_function = None

    def set_channels(self, channels: str):
        """
            Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
            self.channels: used to extract the relevant channels from the hdf5 file
            self.c_index_rgb and self.c_index_nir: indices of the RGB (B2, B3, B4) and NIR channels (B8), w.r.t. the
            output of the self.__getitem__() call
        """
        if channels not in CHANNEL_CONFIG:
            raise ValueError(f"Unknown channel configuration `{channels}`. Choose among {CHANNEL_CONFIG} to "
                             "specify `channels`.\n")
        else:
            self.channels = channels
        
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [1, 2, 3]                         # B2, B3, B4
        else:
            # self.channels in ['all', 'all-mask']
            self.num_channels = 13
            self.c_index_rgb = torch.Tensor([3, 2, 1]).long()
            self.s2_channels = list(np.arange(13))               # all 13 bands

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [7]                              # B8
        elif 'all' in channels:
            self.c_index_nir = torch.Tensor([7]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        if '-mask' in self.channels:
            # self.channels in ['bgr-mask', 'bgr-nir-mask', 'all-mask']
            self.num_channels += 1

        if self.use_SAR:
            self.num_channels += 2


    def subsample_sequence(self):
        return NotImplementedError


    def generate_masks(self):
        return NotImplementedError


    def __getitem__(self, item: int) -> Dict[str, Union[np.ndarray, str, List[str]]]:
        """
        Retrieves an item from the dataset.

        Parameters:
        - item (int): Index of the item to retrieve.

        Returns:
        - Dict[str, Union[np.ndarray, str, List[str]]]: A dictionary containing the data, name, masks, and dates.
        """
        patch_data = self.patches_dataset.iloc[item]
        patch_window = Window(*patch_data.window)
        patch_S2_array = SentinelDataProcessor.read_MS(
            patch_data.files[0], patch_window
        )  # Extraction données S2

        (
            patch_S2_curated,
            dates_S2_curated,
            cloud_masks,
        ) = SentinelDataProcessor.extract_and_transform_S2(
            patch_S2_array,
            self.dates_dict[patch_data.mgrs25]["S2"],
            S2_channels_selected=self.s2_channels,
        )

        if self.use_SAR:
            # Appariement des dates S1 ASC ou DESC les plus proches
            (
                dates_S1_curated,
                index_S1_curated,
                orbit_type,
            ) = SentinelDataProcessor.get_pairedS1(
                dates_S2_curated,
                self.dates_dict[patch_data.mgrs25]["S1"]["ASC"],
                self.dates_dict[patch_data.mgrs25]["S1"]["DESC"],
            )
            path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
            patch_S1_array = SentinelDataProcessor.read_SAR(
                path_S1, patch_window
            )  # Extraction données S1
            bands_S1 = [patch_S1_array[t_index] for t_index in index_S1_curated]
            patch_S1_curated = np.stack(bands_S1, axis=0)
            
            patch = np.concatenate(
                [
                    SentinelDataProcessor.process_MS(patch_S2_curated),
                    SentinelDataProcessor.process_SAR(patch_S1_curated),
                ],
                axis=-1,
            )
        else:
            patch = SentinelDataProcessor.process_MS(patch_S2_curated)

        # Partie sur le sampling ou trim de dates    
        if t_sampled is None:
            t_sampled, masks_valid_obs = self.subsample_sequence(patch, seq_length=patch.shape[0])
        else:
            masks_valid_obs = torch.ones(len(t_sampled), )        
        
        frames_input = patch[t_sampled, :, :, :].clone()
        frames_target = patch[t_sampled, :, :, :].clone()

        # Partie sur la génération de masks
        frames_input, masks = self.generate_masks(sample, frames_input, cloud_masks)

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = encodings.get_position_for_positional_encoding(dates_S2_curated, 'day-within-sequence')
        # Get positions for positional encoding
        position_days = encodings.get_position_for_positional_encoding(dates_S2_curated, self.pe_strategy)

        # Assemble output
        out = {
            'x': frames_input,  # (synthetically masked) S2 satellite image time series, (T x C x H x W), optionally including S1 bands
            'y': frames_target,  # observed/target satellite image time series, (T x C x H x W)
            'masks': masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            'masks_valid_obs': masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            'position_days': position_days,
            'days': days,    # temporal sampling, number of days since the first observation in the sequence, (T, )
            'sample_index': item,
            'filepath': self.patches_dataset.iloc[item].files,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'S2_dates' : dates_S2_curated,
        }
        if self.use_SAR:
            out['S1_dates'] = dates_S1_curated
        return out