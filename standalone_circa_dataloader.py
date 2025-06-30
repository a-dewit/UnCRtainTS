# minimal Python script to demonstrate utilizing the pyTorch data loader for SEN12MS-CR and SEN12MS-CR-TS

import os
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.append('/media/DATA/ADeWit/3STR/code/UnCRtainTS')
from dataloader_CIRCA.datasets.CIRCA_dataset import CircaPatchDataSet
from dataloader_CIRCA.tools.collate_functions import circa_collate_fn 

# Résoudre l'erreur CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

print(f"\n🔍 Test du dataloader CIRCA...")

try:
    circa_path = Path('/media/store-dai/projets/pac/3str/EXP_2')
    data_raster = circa_path / 'Data_Raster'
    input_t = 3
    print(f"Chemin CIRCA: {circa_path}")
    
    # Vérifier les chemins
    optique_path = data_raster / 'optique_dataset'
    radar_path = data_raster / 'radar_dataset_v4'
    
    if not circa_path.exists():
        print(f"❌ Le répertoire {circa_path} n'existe pas")
    
    # Essayer de créer le dataset CIRCA
    ds = CircaPatchDataSet(
        data_optique=optique_path,
        data_radar=radar_path,
        patch_size=256,
        overlap=0,
    )
    for i in range(100):
        sample = ds[i]
        for key, value in sample.items():
            if hasattr(value, 'shape') and value.shape == (53, 256, 256, 4):
                print(f'    shape: {value.shape}')
                print(sample['name'])
            

    """
    print('\n🔍 Analyse du sample:')
    sample = ds[0]
    for key, value in sample.items():
        print(f'  {key}: type={type(value)}, dtype={getattr(value, "dtype", "N/A")}')
        if hasattr(value, 'shape'):
            print(f'    shape: {value.shape}')
        if isinstance(value, np.ndarray) and value.dtype.kind == 'U':
            print(f'    ⚠️ STRING UNICODE DÉTECTÉ: {value[:3] if len(value) > 3 else value}')


    print(f"\n✅ Dataset CIRCA créé avec {len(ds)} échantillons")
    
    # Test d'un échantillon avec fonction de collation personnalisée
    print("\n🧪 Test avec collate_fn personnalisée...")
    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=2,  # Tester avec plusieurs échantillons
        shuffle=False, 
        num_workers=0,
        collate_fn=circa_collate_fn  # Utiliser notre fonction personnalisée
    )
    
    for pdx, patch in enumerate(dataloader):
        print(f"\n📦 CIRCA Batch {pdx}:")
        print(f"  Type: {type(patch)}")
        print(f"  Keys: {list(patch.keys())}")

        # Analyser la structure
        if isinstance(patch, dict):
            for key, value in patch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    - {key}: {value.shape}")
                else:
                    print(f"    - {key}: {type(value)}")
        
        if pdx  >= 2:
            break
    """
except ImportError as e:
    print(f"❌ Erreur d'import CIRCA: {e}")
except Exception as e:
    print(f"❌ Erreur CIRCA: {e}")
    import traceback
    traceback.print_exc()

