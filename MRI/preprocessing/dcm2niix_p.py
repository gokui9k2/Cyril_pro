import os
import tempfile
import subprocess
import math 
from google.cloud import storage
import pandas as pd 
import gcsfs
from data_cleaning import TAG_MAPPING

LOGS_FOLDER = "logs/"
MOUNT_POINT = "/mnt/gcs" 

TASK_INDEX = os.environ.get('CLOUD_RUN_TASK_INDEX', '0')

LOG_FAIL_FILE = f"failures_task_{TASK_INDEX}.txt"
LOG_MULTI_FILE = f"multiple_volumes_task_{TASK_INDEX}.txt"

REQUIRED_COLUMNS = list(TAG_MAPPING.keys()) 
if "dicom_address" not in REQUIRED_COLUMNS:
    REQUIRED_COLUMNS.append("dicom_address")

def load_full_compass(bucket_name, clean_folder="clean_data/"):
    """
    Loading the parquet files from the GCS bucket and cleaning the dicom_address paths.
    """
    fs = gcsfs.GCSFileSystem()
    file_path = f"{bucket_name}/{clean_folder}*boussole*.parquet"
    files = fs.glob(file_path)
    
    if not files:
        print("Empty folder no files to load")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            # Loading only required columns to save memory
            df_part = pd.read_parquet(f"gs://{f}", columns=REQUIRED_COLUMNS)
            dfs.append(df_part)
        except Exception as e:
            print(f"Error {f}: {e}")
            continue
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        prefix_gcs = f"gs://{bucket_name}/"
        # Cleaning the dicom_address paths to local mount point
        full_df['dicom_address'] = full_df['dicom_address'].str.replace(prefix_gcs, f"{MOUNT_POINT}/", regex=False)
        
        return full_df
    else:
        return pd.DataFrame()
    
def upload_logs_to_bucket(bucket_name):
    """
    Upload the dynamic logs files to the GCP bucket
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for log_file in [LOG_FAIL_FILE, LOG_MULTI_FILE]:
        if os.path.exists(log_file):
       
            blob_name = f"logs/{log_file}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(log_file)
            print(f"Successfully uploaded log: {blob_name}")

def process_single_volume(dce_id, df_subset, bucket_name, output_folder="nifti_data/"):
    """
    This function transform a single DCE series from DICOM to NIfTI and uploads it to GCS.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_virtual_dir = os.path.join(temp_dir, "in")
        nifti_dir = os.path.join(temp_dir, "out")
        os.makedirs(dicom_virtual_dir)
        os.makedirs(nifti_dir)

        files_linked = 0
        
        # Create the symbolic links to DICOM files
        for _, row in df_subset.iterrows():
            local_source = row['dicom_address']
            
            if os.path.exists(local_source):
                filename = os.path.basename(local_source)
                link_dest = os.path.join(dicom_virtual_dir, filename)
                os.symlink(local_source, link_dest)
                files_linked += 1
            else:
                if files_linked == 0: 
                    print(f"⚠️ File not found : {local_source}")

        if files_linked == 0:
            print(f"⏭️ SKIP {dce_id}: No DICOM files found.")
            
            with open(LOG_FAIL_FILE, "a") as f:
                f.write(f"{dce_id},NO_FILES_FOUND\n")
            return

        # Conversion to NIfTI using dcm2niix
        cmd = ["dcm2niix", "-z", "y", "-f", str(dce_id), "-o", nifti_dir, dicom_virtual_dir]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        generated = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
        
        if generated:
            # Anomaly handling : multiple files generated
            if len(generated) > 1:
                with open(LOG_MULTI_FILE, "a") as f:
                    f.write(f"{dce_id},{len(generated)}_files_generated\n")

            final_name = max(generated, key=lambda f: os.path.getsize(os.path.join(nifti_dir, f)))
            local_path = os.path.join(nifti_dir, final_name)
            
            # Upload
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"{output_folder}{dce_id}.nii.gz")
                blob.upload_from_filename(local_path)
                print(f"✅ OK {dce_id}.")
            except Exception as e:
                print(f"❌ ERROR upload {dce_id}: {e}")
                with open(LOG_FAIL_FILE, "a") as f:
                    f.write(f"{dce_id},UPLOAD_ERROR\n")
        else:
            print(f"❌ FAIL {dce_id}")
            with open(LOG_FAIL_FILE, "a") as f:
                f.write(f"{dce_id},CONVERSION_FAILED\n")

def get_my_sharded_workload(df, id_column='DCE_ID'):
    """
    This function divides the workload among multiple Cloud Run tasks based on unique IDs
    """
    task_index = int(os.environ.get('CLOUD_RUN_TASK_INDEX', 0))
    task_count = int(os.environ.get('CLOUD_RUN_TASK_COUNT', 1)) 
    
    all_unique_ids = df[id_column].unique().tolist()
    total_items = len(all_unique_ids)
    
    if total_items == 0:
        return [], pd.DataFrame()

    chunk_size = math.ceil(total_items / task_count) 
    start_idx = task_index * chunk_size           
    end_idx = start_idx + chunk_size              
    
    my_ids = all_unique_ids[start_idx:end_idx]
    my_df = df[df[id_column].isin(my_ids)]
    
    return my_ids, my_df