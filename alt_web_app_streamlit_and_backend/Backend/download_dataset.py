import os
import gdown


MOVIES_FILE_ID="1lzwaZK9u9q1TVrLDNORhh8wvY_Pr0n_S"  # Use the env file
RATINGS_FILE_ID ="1Ty37bjeAgKm1UUk3OedD18qCPEjxxTZj"
TAGS_FILE_ID ="12a6upqnERClrGvmrPyEeLUmuNpEAm4IW"
LINKS_FILE_ID="1enPpNPe_TckzQh2YIXWPJwN8MXsTaC36"

def download_file_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using its file ID."""
    if not file_id:
        raise ValueError(f"Invalid file ID: {file_id}")  # Handle missing file ID

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def get_dataset_paths():
    """Retrieve dataset paths and download files if not present. *Uncomment down code for .env* """
    # MOVIES_FILE_ID = os.getenv("MOVIES_FILE_ID")
    # RATINGS_FILE_ID = os.getenv("RATINGS_FILE_ID")
    # TAGS_FILE_ID = os.getenv("TAGS_FILE_ID")

    # if not MOVIES_FILE_ID or not RATINGS_FILE_ID or not TAGS_FILE_ID:
    #     raise EnvironmentError("One or more dataset file IDs are missing in environment variables.")

    DATASET_DIR = "dataset"
    os.makedirs(DATASET_DIR, exist_ok=True)

    movies_path = os.path.join(DATASET_DIR, "movies.csv")
    ratings_path = os.path.join(DATASET_DIR, "ratings.csv")
    tags_path = os.path.join(DATASET_DIR, "tags.csv")

    # Download datasets if they do not exist
    if not os.path.exists(movies_path):
        download_file_from_gdrive(MOVIES_FILE_ID, movies_path)

    if not os.path.exists(ratings_path):
        download_file_from_gdrive(RATINGS_FILE_ID, ratings_path)
    
    if not os.path.exists(tags_path):
        download_file_from_gdrive(TAGS_FILE_ID, tags_path)

    return [movies_path, ratings_path, tags_path]