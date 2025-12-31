import kagglehub
import shutil
from pathlib import Path
import tempfile

def download_from_kaggle(target_dir:str="dataset"):
    target_path = Path.cwd() / target_dir
    # step 1: check if the dataset already exists
    if target_path.exists() and any(target_path.iterdir()):
        print(f" Dataset already exists at: {target_path.resolve()}")
        return target_path
    #step 2: downloading dataset from kaggle  
    print(" Downloading dataset from kagglehub...")
    cache_path=kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Data downloaded into the cache path:{cache_path} ")
    # step 3: copying dataset into current working directory
    print(f" Copying dataset to: {target_path.resolve()}")
    shutil.copytree(src=Path(cache_path),dst=target_path,dirs_exist_ok=True)
    cache_root = Path.home() / ".cache" / "kagglehub"
    if cache_root.exists():
        print(f" Clearing Kagglehub cache: {cache_root}")
        shutil.rmtree(cache_root, ignore_errors=True)
    else:
        print(" Kagglehub cache folder not found, nothing to clear.")
    print("done!")
    return target_path
