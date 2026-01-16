import io
import os
import zipfile
import pandas as pd
import pydicom
import tempfile
import time
import math
import concurrent.futures
from google.cloud import storage

SOURCE_BUCKET_NAME = "mri-dl"       
TARGET_BUCKET_NAME = "playground-dl"      

RAW_FOLDER = "dicoms/"              
CLEAN_FOLDER = "clean_data/"        
MAX_WORKERS = 4 

def safe_convert(x):
    try:
        return float(x)
    except ValueError:
        return str(x)

def get_all_metadata_raw(ds):
    data = {}
    for elem in ds:
        if elem.tag.group == 0x7FE0: continue
        if elem.VR in ['OB', 'OW', 'OF', 'OD', 'UN', 'SQ']: continue 
        
        tag_hex = f"{elem.tag.group:04X}{elem.tag.element:04X}"
        
        try:
            val = elem.value
            if val is None: continue

            if isinstance(val, (pydicom.multival.MultiValue, list, tuple)):

                clean_list = [safe_convert(x) for x in val]
                data[tag_hex] = str(clean_list)
                
            else:
                data[tag_hex] = str(safe_convert(val))
                
        except Exception:
            continue
            
    return data

def process_and_extract_zip(blob_name):
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(SOURCE_BUCKET_NAME)
    target_bucket = storage_client.bucket(TARGET_BUCKET_NAME)
    
    zip_folder_name = blob_name.replace('.zip', '')
    extracted_records = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_path = os.path.join(temp_dir, "temp.zip")
        try:
            blob = source_bucket.blob(blob_name)
            blob.download_to_filename(temp_zip_path)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as z:
                for internal_path in z.namelist():
                    if internal_path.endswith('/') or '__MACOSX' in internal_path: continue
                    try:
                        with z.open(internal_path) as dcm_file:
                            raw_bytes = dcm_file.read()
                            try:
                                ds = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True)
                            except: continue 
                            
                            target_path = f"{RAW_FOLDER}{zip_folder_name}/{internal_path}"
                            new_blob = target_bucket.blob(target_path)
                            new_blob.upload_from_string(raw_bytes, content_type="application/dicom")
                            

                            meta_dict = get_all_metadata_raw(ds)
                            record = {"dicom_address": f"gs://{TARGET_BUCKET_NAME}/{target_path}","source_zip": blob_name,"internal_path": internal_path}
                            record.update(meta_dict)
                            extracted_records.append(record)
                    except: pass
        except Exception as e:
            print(f"Error ZIP {blob_name}: {e}")
            return []

    return extracted_records

def main():
    t0 = time.time()
    client = storage.Client()
    source_bucket = client.bucket(SOURCE_BUCKET_NAME)

    task_index = int(os.environ.get('CLOUD_RUN_TASK_INDEX', 0))
    task_count = int(os.environ.get('CLOUD_RUN_TASK_COUNT', 1))

    blobs = list(source_bucket.list_blobs())
    all_zip_names = [b.name for b in blobs if b.name.endswith(".zip")]
    
    all_zip_names = all_zip_names

    chunk_size = math.ceil(len(all_zip_names) / task_count)
    
    chunk_size = math.ceil(len(all_zip_names) / task_count)
    start_idx = task_index * chunk_size
    end_idx = start_idx + chunk_size
    my_zips = all_zip_names[start_idx:end_idx]
    
    if not my_zips:
        print("Empty workload for this task")
        return

    all_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_zip = {executor.submit(process_and_extract_zip, name): name for name in my_zips}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_zip)):
            try:
                res = future.result()
                if res: all_records.extend(res)
                if (i+1) % 5 == 0: 
                    print(f"Worker {task_index} : {i+1}/{len(my_zips)} ZIPs processed")
            except: pass

    if all_records:
        output_filename = f"boussole_part_{task_index}.parquet"

        try:
            df = pd.DataFrame(all_records)
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)

            df.to_parquet(output_filename, index=False)
            
            target_bucket = client.bucket(TARGET_BUCKET_NAME)
            target_bucket.blob(f"{CLEAN_FOLDER}{output_filename}").upload_from_filename(output_filename)
            print(f"Sucessfully saved")
        except Exception as e:
            print(f"Error occurred while saving : {e}")
    else:
        print("No records extracted")

if __name__ == "__main__":
    main()