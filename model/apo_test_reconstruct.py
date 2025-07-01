"""
Script for image reconstruction inference with pre-trained models
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import pprint
import sys
from pathlib import Path

import torch
from parse_args import create_parser

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from src import utils
from src.model_utils import get_model, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from train_reconstruct import (
    import_from_path,
    iterate,
    prepare_output,
    save_results,
    seed_packages,
)

from data.apo_dataloader import CIRCA_from_HDF5

parser = create_parser(mode="test")
test_config = parser.parse_args()

# grab the PID so we can look it up in the logged config for server-side process management
test_config.pid = os.getpid()

# load previous config from training directories

# if no custom path to config file is passed, try fetching config file at default location
conf_path = (
    os.path.join(
        dirname, test_config.weight_folder, test_config.experiment_name, "conf.json"
    )
    if not test_config.load_config
    else test_config.load_config
)
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = [
            "pid",
            "device",
            "resume_at",
            "trained_checkp",
            "res_dir",
            "weight_folder",
            "root1",
            "root2",
            "root3",
            "max_samples_count",
            "batch_size",
            "display_step",
            "plot_every",
            "export_every",
            "input_t",
            "region",
            "min_cov",
            "max_cov",
        ]
        conf_dict = {
            key: val for key, val in model_config.items() if key not in no_overwrite
        }
        for key, val in vars(test_config).items():
            if key in no_overwrite:
                conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else:
    config = test_config  # otherwise, keep passed flags without any overwriting
config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

if config.pretrain:
    config.batch_size = 32

experime_dir = os.path.join(config.res_dir, config.experiment_name)
if not os.path.exists(experime_dir):
    os.makedirs(experime_dir)
with open(os.path.join(experime_dir, "conf.json"), "w") as file:
    file.write(json.dumps(vars(config), indent=4))

# seed everything
seed_packages(config.rdm_seed)
#if __name__ == "__main__":
#    pprint.pprint(config)

# instantiate tensorboard logger
writer = SummaryWriter(os.path.join(config.res_dir, config.experiment_name))


def main(config):
    device = torch.device(config.device)
    prepare_output(config)

    model = get_model(config)
    model = model.to(device)
    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    #print(model)

    # get data loader
    #path_dataset_circa = Path("/media/DATA/ADeWit/3STR/dataset")
    #hdf5_file = path_dataset_circa / "toy_circa_ligth_0.5.hdf5"
    hdf5_file = config.hdf5_file
    # Import des données depuis un fichier hdf5
    dataset = CIRCA_from_HDF5(
        hdf5_file=hdf5_file,
        phase="all",
        shuffle=False,
        channels="all",
    )

    #dt_test = torch.utils.data.Subset(dataset, range(0, min(config.max_samples_count, len(dataset))))
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False
    )


    print('DT TEST')
    print(dataset[0]['input']['S2 TD'])
    
    print('TEST LOADER')
    print(test_loader.dataset[0]['input']['S2 TD'])


    # Load weights
    ckpt_n = f"_epoch_{config.resume_at}" if config.resume_at > 0 else ""
    load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")

    # Inference
    print("Testing . . .")
    model.eval()

    _, test_img_metrics = iterate(
        model,
        data_loader=test_loader,
        config=config,
        writer=writer,
        mode="test",
        epoch=1,
        device=device,
    )
    print(f"\nTest image metrics: {test_img_metrics}")

    save_results(
        test_img_metrics,
        os.path.join(config.res_dir, config.experiment_name),
        split="test",
    )
    print(
        f"\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}"
    )


if __name__ == "__main__":
    main(config)
    sys.exit()
