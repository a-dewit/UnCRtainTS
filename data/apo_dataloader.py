import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import datetime as dt

import h5py
from torch.utils.data import Dataset

from data.constants import MGRSC_SPLITS

MAX_SEQ_LENGTH = 30
MIN_SEQ_LENGTH = 5
SEED = 42


# Fonction utiles
def str2date(date_string: str) -> dt.date:
    """Converts a date in string format to datetime format."""
    return dt.datetime.strptime(date_string, "%Y%m%d").date()

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


class CIRCA_from_HDF5(Dataset):
    """
    Dataset qui importe les données CIRCA depuis un fichier HDF5.
    """

    def __init__(
        self,
        phase: str = "all",
        hdf5_file: Optional[Union[str, Path]] = None,
        shuffle: bool = False,
        include_S1: bool = True,
        channels: Optional[str] = "all",
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.include_S1 = include_S1
        self.rng = np.random.default_rng(seed=SEED)  # Def random number generator
        self.hdf5_file, self.patches_dataset = self.setup_hdf5_file(hdf5_file)
        self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)

    def setup_hdf5_file(self, path_file):
        if Path(path_file).exists():
            f = h5py.File(path_file, "r", libver="latest", swmr=True)
            patches_dataset = self.list_files_in_hdf5(f)
            patches_dataset = self.splits_samples(patches_dataset, self.phase)
        else:
            raise FileNotFoundError(f"HDF5 file {path_file} does not exist.")

        return f, patches_dataset

    def setup_channels(self, channels: str):
        if channels == "all":
            num_channels = 10
            c_index_rgb = torch.Tensor([2, 1, 0]).long()
            c_index_nir = torch.Tensor([6]).long() #index 6 : B8
            s2_channels = list(np.arange(10))
        elif channels == "bgr-nir":
            num_channels = 4
            c_index_rgb = torch.Tensor([2, 1, 0]).long()
            c_index_nir = torch.Tensor([6]).long()
            s2_channels = [0, 1, 2, 6]
        else:
            raise ValueError(f"Channels {channels} not recognized. Use 'all' or 'bgr-nir'.")

        if self.include_S1:
            num_channels += 4

        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def list_files_in_hdf5(self, hdf5_file: Union[str, Path]) -> pd.DataFrame:
        """
        list all files in a given HDF5 file and return their details in a DataFrame.

        This method iterates through the hierarchical structure of the HDF5 filetrue
        extracting the MGRS levels, MGRS25 levels, and window names, and compiles
        them into a pandas DataFrame.

        Args:
            hdf5_file (Union[str, Path]): The path to the HDF5 file or the file object itself.

        Returns:
            pd.DataFrame: A DataFrame containing the MGRS levels, MGRS25 levels, and window names.
        """
        patches_dataset = pd.DataFrame(columns=["mgrs", "mgrs25", "window"])
        for mgrs_level, mgrsc_list in hdf5_file.items():
            for mgrsc_level, mgrsc_data in mgrsc_list.items():
                for window in mgrsc_data.keys():
                    data = {
                        "mgrs": [mgrs_level],
                        "mgrs25": [mgrsc_level],
                        "window": [window],
                    }
                    patches_dataset = pd.concat([patches_dataset, pd.DataFrame(data)], ignore_index=True)
        if self.shuffle:
            patches_dataset = patches_dataset.sample(frac=1).reset_index(drop=True)
        return patches_dataset

    def splits_samples(self, patches_dataset: pd.DataFrame, phase: str) -> pd.DataFrame:
        if phase is not None:
            if phase in MGRSC_SPLITS:
                patches_dataset = patches_dataset[patches_dataset["mgrs25"].isin(MGRSC_SPLITS[self.phase])].reset_index(
                    drop=True
                )
            elif phase == "train+val":
                patches_dataset = patches_dataset[
                    patches_dataset["mgrs25"].isin(MGRSC_SPLITS["train"] + MGRSC_SPLITS["val"])
                ].reset_index(drop=True)
            elif phase == "all":
                pass
            else:
                raise ValueError(f"Phase {phase} not recognized. Use 'train', 'val', 'train+val', or 'all'.")
            return patches_dataset
        else:
            raise ValueError("Phase is not defined. Use 'train', 'val', 'train+val', or 'all'.")

    def decode_dates(self, dates):
        return np.array([d.decode("utf-8") for d in dates])
        #np.asanyarray([el.decode("utf-8") for el in dates])

    def format_item(self, sample: dict):
        """
        Passage de T * C * H * W au bon format pour le modèle.
        Garde les 10 premières dates

        """
        S1_LAUNCH = str2date("20140403")
        s1_td = [(str2date(date) - S1_LAUNCH).days for date in sample["S1"]["S1_dates"][:10]]
        s1_td = recursive_todevice(torch.tensor(s1_td), 'cuda')
        s2_td = [(str2date(date) - S1_LAUNCH).days for date in sample["S1"]["S1_dates"][:10]]
        S2_td  = recursive_todevice(torch.tensor(s2_td), 'cuda')

        s1_dates = torch.tensor([date.astype(np.int64) for date in sample["S1"]["S1_dates"][:10]])
        s2_dates = torch.tensor([date.astype(np.int64) for date in sample["S2"]["S2_dates"][:10]])

        return {
            "input": {
                "S1": torch.from_numpy(sample["S1"]["S1"][:10,:,:,:].astype(np.float32)),  # T * C * H * W
                "S1_dates": s1_dates,
                "S2": torch.from_numpy(sample["S2"]["S2"][:10,:,:,:].astype(np.float32)),  # T * C * H * W
                "S2_dates": s2_dates,
                #np.array([date.astype(np.int64) for date in sample["S2"]["S2_dates"][:10]]),
                "S1 TD": s1_td,
                "S2 TD": S2_td, 
                "cloud_mask": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_mask"][:10], axis=1).astype(np.float32)),
                "cloud_prob": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_prob"][:10], axis=1).astype(np.float32)),
                "idx_cloudy_frames": torch.from_numpy(sample["idx_cloudy_frames"][:10]),
                "idx_good_frames": torch.from_numpy(sample["idx_good_frames"][:10]),
                "idx_impaired_frames": torch.from_numpy(sample["idx_impaired_frames"][:10]),
                "valid_obs": torch.from_numpy(sample["valid_obs"][:10])
            }
        }

    def etl_item(self, item: int) -> dict[str, Union[np.ndarray, list[str]]]:
        row = self.patches_dataset.iloc[item]
        patch = self.hdf5_file[f"{row.mgrs}/{row.mgrs25}/{row.window}"]
        sample = {
            "S1": {
                "S1": patch["S1/S1"][:],  # T * C * H * W
                "S1_dates": self.decode_dates(patch["S1/S1_dates"][:]),
            },
            "S2": {
                "S2": patch["S2/S2"][:],  # T * C * H * W
                "S2_dates": self.decode_dates(patch["S2/S2_dates"][:]),
                "cloud_mask": patch["S2/cloud_mask"][:],  # T * C * H * W
                "cloud_prob": patch["S2/cloud_prob"][:],  # T * C * H * W
            },
            "idx_cloudy_frames": patch["idx_cloudy_frames"][:],
            "idx_good_frames": patch["idx_good_frames"][:],
            "idx_impaired_frames": patch["idx_impaired_frames"][:],
            "valid_obs": patch["valid_obs"][:],
        }
        t = self.format_item(sample)
        for k in t['input'].keys():
            if (k == 'S1_dates') or (k=='S1 TD'):
                print(k, t['input'][k][0])
        return self.format_item(sample)

    def __getitem__(
        self,
        item: int,
    ) -> dict[str, torch.Tensor]:
        patch_data = self.etl_item(item=item)
        # Select the correct channels
        if self.num_channels != patch_data["input"]["S2"].shape[1]:
            patch_data["input"]["S2"] = patch_data["input"]["S2"][:, self.s2_channels, :, :]
        return patch_data

    def __len__(self):
        return len(self.patches_dataset)

######################################################################################
######################################################################################
######################################################################################

if __name__ == "__main__":

    path_dataset_circa = Path("/media/DATA/ADeWit/3STR/dataset")
    hdf5_file = path_dataset_circa / "toy_circa_ligth_0.5.hdf5"

    # Import des données depuis un fichier hdf5
    dataset = CIRCA_from_HDF5(
        hdf5_file=hdf5_file,
        phase="all",
        shuffle=False,
        channels="all",
    )
    print(dataset)
    sample = next(iter(dataset))
    print(sample.keys())
    print(dataset.__getitem__(0))
