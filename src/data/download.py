import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.utils.config import RAW_DATA_DIR, DATASET_NAME, DATASET_URL, create_directories


def download_movielens_100k() -> Path:
    """
    Download and extract the MovieLens 100K dataset if not already present.
    Returns the extracted dataset directory path.
    """
    create_directories()

    zip_path = RAW_DATA_DIR / f"{DATASET_NAME}.zip"
    extract_path = RAW_DATA_DIR / DATASET_NAME

    if not zip_path.exists():
        print(f"Downloading dataset from: {DATASET_URL}")
        urlretrieve(DATASET_URL, zip_path)
        print("Download complete.")

    if not extract_path.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Extraction complete.")

    return extract_path


if __name__ == "__main__":
    path = download_movielens_100k()
    print(f"Dataset available at: {path}")