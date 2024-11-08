from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import torchaudio
import torch
import pandas as pd


def generate_VoxCeleb_metadata(voxceleb_root, save_path):
    filepaths = []
    for file in Path(voxceleb_root).rglob("*.wav"):
        filepaths.append(file)

    data = []
    for file in tqdm(filepaths, total=len(filepaths)):
        speaker_id, chapter_id, utterance_id = str(file).split("/")[-3:]
        utterance_id = str(Path(utterance_id).stem)
        speaker_id = int(speaker_id.strip("id_"))
        info = torchaudio.info(file)
        sample_rate = info.sample_rate
        num_frames = info.num_frames
        bits_per_sample = info.bits_per_sample
        encoding = info.encoding
        num_channels = info.num_channels
        duration_seconds = num_frames / sample_rate
        relative_file_path = Path(file).relative_to(voxceleb_root)
        data.append(
            [relative_file_path, sample_rate, num_frames, bits_per_sample, encoding, num_channels, duration_seconds,
             speaker_id, chapter_id, utterance_id])

    header_names = ["relative_path", "sample_rate", "num_frames", "bits_per_sample", "encoding", "num_channels",
                    "duration_seconds", "speaker_id", "chapter_id", "utterance_id"]
    data_df = pd.DataFrame(data, columns=header_names)
    data_df.to_csv(save_path)


def generate_VoxCeleb_speech_only_metadata(root_speech_only, save_path):
    filepaths = []
    for file in Path(root_speech_only).rglob("*.pt"):
        filepaths.append(file)

    data = []
    for file in tqdm(filepaths, total=len(filepaths)):
        speaker_id, chapter_id, utterance_id = str(file).split("/")[-3:]
        utterance_id = str(Path(utterance_id).stem)
        speaker_id = int(speaker_id.strip("id_"))
        features = torch.load(file)
        num_frames = features.size(-1)
        duration_seconds = num_frames * 0.01
        relative_file_path = Path(file).relative_to(root_speech_only)
        data.append([relative_file_path, num_frames, duration_seconds,
                     speaker_id, chapter_id, utterance_id])

    header_names = ["relative_path", "num_frames", "duration_seconds", "speaker_id", "chapter_id", "utterance_id"]
    data_df = pd.DataFrame(data, columns=header_names)
    data_df.to_csv(save_path)


def generate_LibriSpeech_metadata(voxceleb_root, save_path):
    filepaths = []
    for file in Path(voxceleb_root).rglob("*.flac"):
        filepaths.append(file)

    data = []
    for file in tqdm(filepaths, total=len(filepaths)):
        speaker_id, chapter_id, utterance_id = str(Path(file).stem).split("-")
        speaker_id = int(speaker_id)
        info = torchaudio.info(file)
        sample_rate = info.sample_rate
        num_frames = info.num_frames
        bits_per_sample = info.bits_per_sample
        encoding = info.encoding
        num_channels = info.num_channels
        duration_seconds = num_frames / sample_rate
        relative_file_path = Path(file).relative_to(voxceleb_root)
        data.append(
            [relative_file_path, sample_rate, num_frames, bits_per_sample, encoding, num_channels, duration_seconds,
             speaker_id, chapter_id, utterance_id])

    header_names = ["relative_path", "sample_rate", "num_frames", "bits_per_sample", "encoding", "num_channels",
                    "duration_seconds", "speaker_id", "chapter_id", "utterance_id"]
    data_df = pd.DataFrame(data, columns=header_names)
    data_df.to_csv(save_path)


def generate_LibriSpeech_speech_only_metadata(root_speech_only, save_path):
    filepaths = []
    for file in Path(root_speech_only).rglob("*.pt"):
        filepaths.append(file)

    data = []
    for file in tqdm(filepaths, total=len(filepaths)):
        speaker_id, chapter_id, utterance_id = str(Path(file).stem).split("-")
        speaker_id = int(speaker_id)
        features = torch.load(file)
        num_frames = features.size(-1)
        duration_seconds = num_frames * 0.01
        relative_file_path = Path(file).relative_to(root_speech_only)
        data.append([relative_file_path, num_frames, duration_seconds,
                     speaker_id, chapter_id, utterance_id])

    header_names = ["relative_path", "num_frames", "duration_seconds", "speaker_id", "chapter_id", "utterance_id"]
    data_df = pd.DataFrame(data, columns=header_names)
    data_df.to_csv(save_path)


def main(arguments):
    if arguments.data_set == "VoxCeleb":
        if arguments.features:
            generate_VoxCeleb_speech_only_metadata(arguments.root,
                                                   save_path=arguments.save_path)
        else:
            generate_VoxCeleb_metadata(arguments.root, save_path=arguments.save_path)
    elif arguments.data_set == "LibriSpeech":
        if arguments.features:
            generate_LibriSpeech_speech_only_metadata(arguments.root,
                                                      save_path=arguments.save_path)
        else:
            generate_LibriSpeech_metadata(arguments.root, save_path=arguments.save_path)
    else:
        raise ValueError(f"Data set '{arguments.data_set}' not recognized.")


if __name__ == "__main__":
    parser = ArgumentParser("Script for generating VoxCeleb and LibriSpeech metadata.csv for GE2E data loader.")
    parser.add_argument("--data_set", type=str, required=True, help="Name of dataset.")
    parser.add_argument("--features", type=str, required=True, help="Whether the data are precomputed features.")
    parser.add_argument("--root", type=str, required=True, help="Path to file root.")
    parser.add_argument("--save_path", type=str, required=False, help="Path to save file.",
                        default=None)
    args = parser.parse_args()
    main(arguments=args)
