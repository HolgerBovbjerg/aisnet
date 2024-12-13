from pathlib import Path
from logging import getLogger
import os
import shutil

from huggingface_hub import snapshot_download


logger = getLogger(__name__)


def download_binaural_librispeech_split(root: str, subset: str, split: str):
    repo_id = "Holger1997/BinauralLibriSpeech"
    directory_name = "data/BinauralLibriSpeech"
    output_dir = Path(f"{root}")
    if (output_dir / subset / split).exists():
        logger.info("%s already exists, skipping.", str(output_dir / subset / split))
        return
    logger.info("Downloading from data set from HuggingFace - %s: subset=%s, split=%s.",
                repo_id, subset, split)
    output_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{directory_name}/{subset}/{split}/*"],  # Only download files in the specific folder
        local_dir=root,
        repo_type="dataset"
    )

    # Define the path where files are downloaded
    downloaded_dir = output_dir / directory_name / subset / split

    # Define the target path where the files should go
    target_dir = output_dir / subset / split

    # # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move up files
    for item in downloaded_dir.iterdir():
        target_item = target_dir / item.name
        shutil.move(str(item), str(target_item))  # Move directories

    # Delete the empty parent directories after moving
    shutil.rmtree(os.path.join(root, "data"))

    logger.info("Extracting tar.gz files...")
    # Extract all tar files in the downloaded directory
    for tar_root, _, files in os.walk(str(target_dir)):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_file_path = os.path.join(tar_root, file)
                # Extract the tar file
                shutil.unpack_archive(tar_file_path, target_dir)
                # Delete the tar file after extraction
                os.remove(tar_file_path)
                logger.info("Deleted %s after extraction", tar_file_path)
    logger.info("Finished download and extraction of data to %s.", str(target_dir))


def prepare_data(config):
    train_data_config = config.data.train
    validation_data_config = config.data.validation
    test_data_config = config.data.test

    # Download speech data
    for data_config in [train_data_config, validation_data_config, test_data_config]:
        for data_set, data_set_config in data_config["data_sets"].items():
            if data_set == "BinauralLibriSpeech":
                root = data_set_config.root
                subset = data_set_config.subset
                splits = data_set_config.splits
                for split in splits:
                    download_binaural_librispeech_split(root=root, subset=subset, split=split)
            else:
                logger.error("Dataset %s not supported", data_set)
