from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.functional import resample


class BinauralLibriSpeechDataset(Dataset):
    _ext_audio = ".flac"

    def __init__(self, root_dir, metadata_filename, sample_rate=16000, split="test-clean"):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            metadata_filename (str): Name of the CSV file containing the metadata (in root folder).
            sample_rate (int): Sampling rate of the audio files.
        """
        self.root_dir = root_dir
        self.split = split
        self.metadata = pd.read_csv(f"{self.root_dir}/{split}/{metadata_filename}").set_index("file_name")
        self.sample_rate = sample_rate
        self.audio_files = sorted(str(p.stem) for p in Path(f"{self.root_dir}/{split}").glob("*/*/*" + self._ext_audio))

    def __getitem__(self, idx):
        """
        Yield one sample at a time: audio (stereo) and corresponding metadata.
        """
        audio_file = self.audio_files[idx]
        # Load the audio file
        speaker_id, chapter_id, utterance_id = audio_file.split("-")
        waveform, sr = torchaudio.load(f"{self.root_dir}/{self.split}/{speaker_id}/{chapter_id}/{audio_file}{self._ext_audio}")

        # Ensure the sampling rate matches
        if sr != self.sample_rate:
            waveform = resample(waveform, sr, self.sample_rate)

        # Extract metadata
        metadata = self.metadata.loc[f"{speaker_id}/{chapter_id}/{audio_file}{self._ext_audio}"].to_dict()
        microphone_positions = torch.tensor(
            [[metadata["microphone_left_x"], metadata["microphone_left_y"], metadata["microphone_left_z"]],
             [metadata["microphone_right_x"], metadata["microphone_right_y"], metadata["microphone_right_z"]]])
        output = {"waveform": waveform}
        output.update({"speaker_id": speaker_id, "chapter_id": chapter_id, "utterance_id": utterance_id,
                       "elevation": metadata["elevation"], "azimuth": metadata["azimuth"],
                       "microphone_positions": microphone_positions, "transcript": metadata["transcript"]})
        return output

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    # Usage example
    root_dir = "/Users/JG96XG/Desktop/data_sets/BinauralLibriSpeech/spherical"
    metadata_filename = "metadata.csv"
    dataset = BinauralLibriSpeechDataset(root_dir=root_dir, metadata_filename=metadata_filename, split="test-other")

    # To load and iterate through the dataset
    for data in dataset:
        print(data["waveform"].size())

    print("done")
