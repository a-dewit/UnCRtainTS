# minimal Python script to demonstrate utilizing the pyTorch data loader for SEN12MS-CR and SEN12MS-CR-TS

import os
import glob
import rasterio
import torch

from data.dataLoader import SEN12MSCR, SEN12MSCRTS

if __name__ == "__main__":
    dataset = "SEN12MS-CR-TS"
    root = "/media/DATA/ADeWit/3STR/dataset" 
    split = "all"  # ROI to sample from, belonging to splits [all | train | val | test]
    input_t = 3  # number of input time points to sample
    import_path = None  # path to importing the suppl. file specifying what time points to load for input and output
    sample_type = "cloudy_cloudfree"  # type of samples returned [cloudy_cloudfree | generic]
    
    print(f"🔧 Configuration:")
    print(f"  - Dataset: {dataset}")
    print(f"  - Root: {root}")
    print(f"  - Split: {split}")
    print(f"  - Input timesteps: {input_t}")
    print(f"  - Sample type: {sample_type}")

    dataset_path = os.path.join(root, "SEN12MSCRTS") 
    # Lister les ROIs disponibles
    print(f"\n📁 Contenu du dataset:")
    roi_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith('ROI'):
            roi_dirs.append(item)
            print(f"  - {item}")
    
    if not roi_dirs: print("❌ Aucun répertoire ROI trouvé")


    """
    # Analyser le premier ROI pour comprendre la structure
    roi_path = os.path.join(dataset_path, roi_dirs[0])
    print(f"\n🔍 Analyse de {roi_dirs[0]}:")
    
    # Lister les timesteps
    timesteps = []
    for item in os.listdir(roi_path):
        if item.isdigit():
            timesteps.append(int(item))
    
    timesteps.sort()
    print(f"  Timesteps disponibles: {len(timesteps)} ({min(timesteps)} à {max(timesteps)})")
    
    # Analyser quelques timesteps
    for t in timesteps[:5]:  # Analyser les 5 premiers
        if t == (8 or 21 or 22 or 32 or 35 or 40 or 61 or 75 or 140):
            pass

        t_path = os.path.join(roi_path, str(t))
        print(f"\n  📅 Timestep {t}:")
        
        s1_path = os.path.join(t_path, "S1")
        s2_path = os.path.join(t_path, "S2")
        
        s1_files = []
        s2_files = []
        
        if os.path.exists(s1_path):
           s1_files = glob.glob(os.path.join(s1_path, '*/*.tif'))
           s1_files.sort()
        
        if os.path.exists(s2_path):
            s2_files = glob.glob(os.path.join(s2_path, '*/*.tif'))
            s2_files.sort()
        
        print(f"    - S1: {len(s1_files)} fichiers")
        print(f"    - S2: {len(s2_files)} fichiers")
        
        if len(s1_files) != len(s2_files):
            print(f"    ⚠️  PROBLÈME: Nombre différent de fichiers S1 ({len(s1_files)}) vs S2 ({len(s2_files)})")
            print(f"    S1 files: {s1_files[:3]}...")
            print(f"    S2 files: {s2_files[:3]}...")
    
        for i, f in enumerate(s1_files):
            with rasterio.open(s1_files[i]) as src:
                s1 = src.read()
                s1_shape = s1.shape
            with rasterio.open(s2_files[i]) as src:
                s2 = src.read()
                s2_shape = s2.shape
            if s1_shape[0] != 2 or s2_shape[0] != 13 :
                print(f"⚠️  PROBLÈME de taille : S1 {s1_shape} et S2 {s2_shape}")
    """

    assert dataset in ["SEN12MS-CR", "SEN12MS-CR-TS"]
    if dataset == "SEN12MS-CR":
        loader = SEN12MSCR(os.path.join(root, "SEN12MSCR"), split=split)
    else:
        try : 
            loader = SEN12MSCRTS(
                os.path.join(root, "SEN12MSCRTS"),
                split=split,
                sample_type=sample_type,
                n_input_samples=input_t,
                import_data_path=import_path,
            )
        
            print(f"✅ Dataset créé avec {len(loader)} échantillons")
            
            # Créer le DataLoader
            dataloader = torch.utils.data.DataLoader(
                loader, batch_size=1, shuffle=False, num_workers=1  #max 8
            )
            print("✅ DataLoader créé")

            sample = next(iter(dataloader))
            print(sample['input'].keys())

            # iterate over split and do some data accessing for demonstration
            for pdx, patch in enumerate(dataloader):
                print(f"Fetching {pdx}. batch of data.")

                input_s1 = patch["input"]["S1"]
                input_s2 = patch["input"]["S2"]
                input_c = sum(patch["input"]["coverage"]) / len(patch["input"]["coverage"])
                output_s2 = patch["target"]["S2"]

                if dataset == "SEN12MS-CR-TS":
                    dates_s1 = patch["input"]["S1 TD"]
                    dates_s2 = patch["input"]["S2 TD"]

                print(f"  - Input S1: {len(input_s1), input_s1[0].shape}")
                print(f"  - Input S2: {len(input_s2), input_s2[0].shape}")
                print(f"  - Target S2: {len(output_s2), output_s2[0].shape}")

                print(input_s2[0][0][0][0])
                print(output_s2[0][0][0][0])

                print(patch["input"]["S2 path"])
                print(patch["target"]["S2 path"])
                        
                if pdx >= 0:  # Tester seulement les 3 premiers
                    break

        except Exception as e:
            print(f"❌ Erreur lors de la création du dataset: {e}")
            import traceback
            traceback.print_exc

        print(":sparkles: Finish")