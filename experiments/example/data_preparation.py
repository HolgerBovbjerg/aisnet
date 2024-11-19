from logging import getLogger
from pathlib import Path
import yaml
import subprocess
import os
import sys
import requests
import tarfile

import pandas as pd
from tqdm import tqdm
from torchaudio.datasets import LIBRISPEECH

from source.datasets.utils import check_database


logger = getLogger(__name__)


def create_symlink(source, destination):
    if source == destination:
        print(f"Source {source} is the same as destination {destination}. Symlink not created.")
    else:
        # Ensure target directory exists
        Path(destination).mkdir(parents=True, exist_ok=True)

        for entry in os.listdir(source):
            source_path = os.path.join(source, entry)
            destination_path = os.path.join(destination, entry)
            try:
                Path(source_path).symlink_to(destination_path)
            except FileExistsError:
                logger.info(f"Symlink for {destination_path} already exists.")

def sync_directories(source, destination, options="-a --info=progress2"):
    """
    Syncs two directories using rsync.

    Parameters:
    - source (str): Path to the source directory.
    - destination (str): Path to the destination directory.
    - options (str): rsync options (default is '-av' for archive mode and verbose output).
    """
    # Normalize paths for comparison
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)

    # Check if source is a subdirectory of destination
    if os.path.commonpath([source, destination]) == destination:
        print(f"Source {source} is a subdirectory of destination {destination}. Skipping rsync.")
        return  # Exit the function without syncing

    # Construct the rsync command
    command = ["rsync"] + options.split() + [source, destination]

    try:
        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read stdout line by line to get overall progress
        for line in process.stdout:
            if "%" in line:
                # Print in the same line by using carriage return
                sys.stdout.write("\r" + line.strip())
                sys.stdout.flush()

        # Wait for the process to complete and get the exit code
        process.wait()

        # Check for errors
        if process.returncode == 0:
            print("\nrsync completed successfully.")
        else:
            print("\nrsync encountered an error.")
            print(process.stderr.read().strip())  # Print error details

    except subprocess.CalledProcessError as e:
        print("Error during rsync:", e.stderr)


def get_dataset_path(datasets, name):
    path = datasets.get(name, None)
    if path is None:
        logger.error(f"Dataset '{name}' not found in database."
                     f"Please update database.yaml")
    return path


def download_librispeech_split(librispeech_root, split):
    if Path(librispeech_root / split).is_dir():
        logger.info(f"Dataset '{split}' already exists in {librispeech_root}. Skipping download...")
        return
    path = check_database("LibriSpeech")
    if path is None:
        raise ValueError("LibriSpeech not found in database.")
    elif path == "download":
        logger.info("Downloading LibriSpeech {}".format(split))
        try:
            Path(librispeech_root).mkdir(exist_ok=True, parents=True)
            LIBRISPEECH(root=librispeech_root, url=split, folder_in_archive="", download=True)
        except Exception as e:
            logger.error("Failed to download LibriSpeech {}: {}".format(split, e))
    elif Path(path).is_dir():
        logger.info(f"Copying LibriSpeech split '{split}' from '{path}' to '{librispeech_root}/{split}'")
        sync_directories(os.path.join(path, split), librispeech_root)
    else:
        logger.error(f"Dataset found in database but the listed dataset path '{path}' is not a directory.")


def download_file(url, dest):
    """Downloads a file from a URL and saves it to the specified destination, showing progress."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True, desc=dest) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    bar.update(len(chunk))  # Update the progress bar
        print(f"\nDownloaded {dest}")
    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error downloading {dest}: {e}")


def extract_tar(file_path, extract_to):
    """Extracts a tar.gz file to the specified directory."""
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted {file_path} to {extract_to}")
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")


def download_musan_data(base_dir='Musan'):
    """Downloads and extracts the Musan noise dataset."""
    if Path(base_dir).is_dir():
        logger.info(f"Dataset 'Musan' already exists in {base_dir}. Skipping download...")
        return

    os.makedirs(base_dir, exist_ok=True)

    # URLs for Musan dataset components
    urls = {'musan.tar.gz': 'https://www.openslr.org/resources/17/musan.tar.gz'}
    path = check_database("Musan")
    if path is None:
        raise ValueError("Musan not found in database.")
    elif path == "download":
        logger.info("Downloading Musan data")
        try:
            Path(base_dir).mkdir(exist_ok=True, parents=True)
            for file_name, url in urls.items():
                file_path = os.path.join(base_dir, file_name)
                download_file(url, file_path)
                extract_tar(file_path, base_dir)
        except Exception as e:
            logger.error("Failed to download Musan: {}".format(e))
    elif Path(path).is_dir():
        logger.info(f"Copying Musan noise from '{path}' to '{base_dir}'")
        sync_directories(path, base_dir)
    else:
        logger.error(f"Dataset found in database but the listed dataset path '{path}' is not a directory.")


def create_speaker_mapping(librispeech_root_dir, split):
    speaker_mapping = {}
    label_counter = 0

    # Construct the split directory path
    split_dir = os.path.join(librispeech_root_dir, split)

    # Check if the split directory exists
    if not (os.path.isdir(split_dir)):
        raise ValueError(f"The specified split directory does not exist: {split_dir}")

    # Iterate through the directories of the specified split
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            if file.endswith('.flac'):  # Check for .flac files
                # Extract speaker id from the file path
                # Assuming the structure: <librispeech_root_dir>/<split>/<speaker_id>/<file_name>
                relative_path = os.path.relpath(os.path.join(root, file), librispeech_root_dir)
                speaker_id = relative_path.split(os.sep)[1]  # Adjust index based on directory structure

                if speaker_id not in speaker_mapping:
                    speaker_mapping[speaker_id] = label_counter
                    label_counter += 1

    return speaker_mapping


def prepare_data(config):
    logger.info("Preparing data")

    train_data_config = config.data.train
    validation_data_config = config.data.validation
    test_data_config = config.data.test

    # Download speech data
    for data_config in [train_data_config, validation_data_config, test_data_config]:
        for data_set, data_set_config in data_config["data_sets"].items():
            if data_set == "LibriSpeech":
                root = data_set_config.root
                splits = data_set_config.splits
                for split in splits:
                    download_librispeech_split(root, split)
            else:
                logger.error(f"Dataset {data_set} not supported")
    # Create speaker mappings
    for data_config in [train_data_config, validation_data_config, test_data_config]:
        for data_set, data_set_config in data_config["data_sets"].items():
            if data_set == "LibriSpeech":
                root = data_set_config.root
                splits = data_set_config.splits
                for split in splits:
                    output_file = os.path.join(root, split, "speaker_mapping.csv")
                    if Path(str(output_file)).is_file():
                        logger.info(f"Skipping {split} speaker mapping as it already exists.")
                    else:
                        speaker_map = create_speaker_mapping(root, split)
                        pd.DataFrame(list(speaker_map.items()),
                                     columns=["Speaker ID", "Class Label"]).to_csv(output_file, index=False)
                        logger.info(f"Created {split} speaker mapping at {output_file}.")
            else:
                logger.error(f"Dataset {data_set} not supported")

    # Download noise data
    download_musan_data(base_dir=os.path.join(config.data_path, 'Musan'))

    return 0
