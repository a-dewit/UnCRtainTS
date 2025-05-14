from pathlib import Path

from dataloader_CIRCA.datasets import CircaPatchDataSet

if __name__ == "__main__":
    store_dai = Path("/home/dl/speillet/Partage/store-dai/")
    path_dataset_CIRCA = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_CIRCA / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_CIRCA / "Data_Raster" / "radar_dataset_v4"
    patch_size = 256
    overlap = 0

    ds = CircaPatchDataSet(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=patch_size,
        overlap=overlap,
    )

    # ds.setup()
    # ds.export_dataset()
    sample = next(iter(ds))
    print(sample.keys())
