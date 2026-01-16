import os
import numpy as np
import pandas as pd
import concurrent.futures
import gcsfs
import pickle # Nécessaire pour sauvegarder les données lourdes
from utils_pre import load_nifti
from image_utils import crop_3D, run_registration_benchmark, ICA_test

# Constants
PROJECT_ID = "mri-dl"
BUCKET_NAME = "playground-dl"
INPUT_PREFIX = "nifti_data"
OUTPUT_FOLDER = "bench_final_v1" 

MAX_WORKERS = 2 # Attention à la RAM sur Cloud Run, 2 c'est bien pour 4GB/8GB

def get_fs():
    return gcsfs.GCSFileSystem(project=PROJECT_ID)

def process_file(gcs_path):
    fname = os.path.basename(gcs_path)
    fs = get_fs()
    
    try:
        # 1. Load & Crop
        data, affine, spacing, header = load_nifti(fs, gcs_path)
        data_crop, bbox, stats_crop = crop_3D(data)

        # 2. Run ICA (Multi-components)
        # On calcule tout
        df_meta_15, heavy_15 = ICA_test(data_crop, fname, n_components=15)
        df_meta_10, heavy_10 = ICA_test(data_crop, fname, n_components=10)
        df_meta_7, heavy_7  = ICA_test(data_crop, fname, n_components=7)

        # Fusion des métadonnées (Léger)
        ICA_df_combined = pd.concat([df_meta_15, df_meta_10, df_meta_7], ignore_index=True)
        
        # --- SAUVEGARDE IMMÉDIATE DES DONNÉES LOURDES (CRITIQUE) ---
        # On ne renvoie pas ça au main, on écrit direct sur GCS pour vider la RAM
        heavy_combined = {"ICA_15": heavy_15, "ICA_10": heavy_10, "ICA_7": heavy_7}
        
        # On crée un fichier pickle par patient
        pkl_path = f"{BUCKET_NAME}/{OUTPUT_FOLDER}/heavy_data/{fname.replace('.nii.gz', '')}_ica.pkl"
        with fs.open(pkl_path, 'wb') as f:
            pickle.dump(heavy_combined, f)
            
        # 3. Registration Benchmark
        reg_results_list = run_registration_benchmark(data_crop, spacing)
        
        # Préparation des résultats de registration
        reg_data_to_return = []
        for reg_res in reg_results_list:
            combined = {"File": fname, "Success": True}
            combined.update(stats_crop) 
            combined.update(reg_res)    
            reg_data_to_return.append(combined)
            
        # On renvoie DEUX choses séparées : les stats de Reg et les stats ICA
        return reg_data_to_return, ICA_df_combined
        
    except Exception as e:
        print(f"Error {fname}: {e}")
        # En cas d'erreur, on renvoie des listes vides ou des marqueurs d'erreur
        return [{"File": fname, "Success": False, "Error": str(e)}], pd.DataFrame()

if __name__ == "__main__":

    # Gestion Cloud Run
    try: 
        idx = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0))
        count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))
    except: idx, count = 0, 1
    
    fs = get_fs()
    # Création du dossier heavy_data si besoin (optionnel avec GCS mais propre)
    # fs.makedirs(f"{BUCKET_NAME}/{OUTPUT_FOLDER}/heavy_data", exist_ok=True)

    all_files = fs.glob(f"{BUCKET_NAME}/{INPUT_PREFIX}/*.nii.gz")
    all_files.sort()
    
    # Subsampling (Attention : assure-toi que c'est ce que tu veux en prod)
    subset_files = all_files[::10]
    
    # Sharding
    my_files = subset_files[idx::count]
    print(f"Processing Started: {len(my_files)} files (Task {idx}/{count})")
    
    final_reg_data = []
    final_ica_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as exc:
        # map renvoie les résultats dans l'ordre
        for result in exc.map(process_file, my_files):
            # result est un tuple (reg_list, ica_df)
            reg_list, ica_df = result
            
            # 1. Accumulation Registration
            if reg_list:
                final_reg_data.extend(reg_list)
            
            # 2. Accumulation ICA Stats
            if not ica_df.empty:
                final_ica_data.append(ica_df)
                
    # --- SAUVEGARDE DES CSV ---
    
    # 1. Sauvegarde Registration
    if final_reg_data:
        df_reg = pd.DataFrame(final_reg_data)
        out_path_reg = f"gs://{BUCKET_NAME}/{OUTPUT_FOLDER}/registration_results_task_{idx}.csv"
        df_reg.to_csv(out_path_reg, index=False)
        print(f"Saved REG CSV: {out_path_reg}")
    else:
        print("No Registration data generated")

    # 2. Sauvegarde ICA Metadata
    if final_ica_data:
        df_ica = pd.concat(final_ica_data, ignore_index=True)
        out_path_ica = f"gs://{BUCKET_NAME}/{OUTPUT_FOLDER}/ica_stats_task_{idx}.csv"
        df_ica.to_csv(out_path_ica, index=False)
        print(f"Saved ICA CSV: {out_path_ica}")
    else:
        print("No ICA data generated")