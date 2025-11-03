import os
import shutil
import logging
import concurrent.futures
from tcia_utils import nbia
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from tqdm import tqdm 

AZURE_CONNECTION_STRING = ...
AZURE_CONTAINER_NAME = ...
manifest_file = '/content/drive/MyDrive/Cancer_data/ACRIN-Contralateral-Breast-MR-Feb-2021-manifest.tcia'
output_dir = '/tmp/tcia_data'
MAX_WORKERS = 10  

logging.basicConfig(level=logging.INFO)

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    print("Connection to Azure Blob Storage successful.")
    print(f"Checking if container '{AZURE_CONTAINER_NAME}' exists...")
    try:
        blob_service_client.create_container(AZURE_CONTAINER_NAME)
        print(f"Container '{AZURE_CONTAINER_NAME}' created successfully.")
    except ResourceExistsError:
        print(f"Container '{AZURE_CONTAINER_NAME}' already exists. Continuing.")
except Exception as e:
    print(f"ERROR: Azure connection or container creation failed. Details: {e}")
    exit()

os.makedirs(output_dir, exist_ok=True)

def process_series(series_uid):
    series_temp_dir = os.path.join(output_dir, series_uid)
    try:
        os.makedirs(series_temp_dir, exist_ok=True)
        formatted_series = [{'SeriesInstanceUID': series_uid}]
        nbia.downloadSeries(series_data=formatted_series, path=series_temp_dir)

        for root, dirs, files in os.walk(series_temp_dir):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                blob_name_in_azure = f"{series_uid}/{file_name}"
                blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name_in_azure)
                
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
        
        return f"Success: {series_uid}"

    except Exception as e:
        return f"Error on series {series_uid}: {e}"

    finally:
        if os.path.exists(series_temp_dir):
            shutil.rmtree(series_temp_dir)


print(f"Reading manifest file: {manifest_file}")
series_uids_list = nbia.manifestToList(manifest_file)

if not series_uids_list:
    print("The manifest file is empty or could not be read.")
else:
    total_series = len(series_uids_list)
    print(f"{total_series} series found. Starting parallel processing with {MAX_WORKERS} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_series, series_uids_list), total=total_series))


    print("Processing complete.")