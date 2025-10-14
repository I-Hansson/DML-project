
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import sys

def ensure_kaggle():
    try:
        import kaggle
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def authenticate_kaggle():
    ensure_kaggle()
    api = KaggleApi()
    api.authenticate()
    return api

def download_dataset(dataset_slug: str):
    api = authenticate_kaggle()
    data_path = Path(os.getcwd()) / '..' / 'data'
    data_path.mkdir(exist_ok=True)
    
    if any(data_path.iterdir()):
        print(f"Data already exists in '{data_path}'. Skipping download.")
    else:
        print(f"Downloading dataset '{dataset_slug}' to '{data_path}' ...")
        api.dataset_download_files(dataset_slug, path=str(data_path), unzip=True)
        print("Download complete!")
    
    return data_path