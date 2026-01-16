import sys
import gc
import pandas as pd 
import concurrent.futures 
from data_cleaning import TAG_MAPPING, detect_dce
from dcm2niix_p import get_my_sharded_workload, process_single_volume, load_full_compass, upload_logs_to_bucket

SOURCE_BUCKET_NAME = "playground-dl"
MAX_THREADS = 2 

if __name__ == "__main__":

    df = load_full_compass(bucket_name=SOURCE_BUCKET_NAME, clean_folder="clean_data/")
    
    if df.empty:
        print("Empty Dataset")
        sys.exit(0)

    df = df.rename(columns=TAG_MAPPING)
    cols_to_keep = list(TAG_MAPPING.values()) + ["dicom_address"]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    
    reduced_df = df[available_cols].copy() 

    # Freeing up memory
    del df
    gc.collect() 

    df_dce = detect_dce(reduced_df)

    if 'DCE_ID' not in df_dce.columns:
        print("Error: DCE_ID column not found after detect_dce")
        sys.exit(1)

    # Collect the workload ID for this specific worker
    my_ids, my_df = get_my_sharded_workload(df_dce, id_column='DCE_ID')
    
    if my_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            
            for dce_id in my_ids:

                df_subset = my_df[my_df["DCE_ID"] == dce_id]
                futures.append(executor.submit(process_single_volume, dce_id, df_subset, SOURCE_BUCKET_NAME))
            
            concurrent.futures.wait(futures)

        print("Sucessfully processed all assigned volumes")
    else:
        print("No DCE volumes assigned to this worker.")

    # Uploading the log files to the GCS bucket
    upload_logs_to_bucket(SOURCE_BUCKET_NAME)
    
    print("End of processing")