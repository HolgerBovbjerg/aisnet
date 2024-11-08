import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio


def get_recording_paths(data_path, target_name):
    # To store only the directory paths where files containing the target_name are found
    recording_paths = set()  # Using a set to avoid duplicate directories

    for root, dirs, files in os.walk(data_path):
        # Check if any file in the directory contains the target_name
        for name in files:
            if target_name in name:
                # Add the directory path (root) to the set and break to avoid multiple adds
                recording_paths.add(root)
                break  # No need to check more files in this directory once we find a match

    return list(recording_paths)


class LOCATADataset(Dataset):
    def __init__(self, locata_root: str, split: str = "dev", task: str = "task1", array_name: str = "eigenmike",
                 transform=None):
        """
        Args:
            data_path (str): Path to the LOCATA dataset folder.
            array_name (str): The name of the microphone array (e.g., dicit, benchmark2, eigenmike, dummy).
            transform (callable, optional): Optional transform to be applied on the waveform.
        """
        self.locata_root = locata_root
        self.split_name = split
        self.array_name = array_name
        self.task = task
        self.transform = transform
        self.data_path = os.path.join(locata_root, split, task)

        # List all recording sessions
        self.recordings = get_recording_paths(self.data_path, array_name)

    def load_array_waveform(self, recording):
        # Load the multichannel waveform for the given array
        audio_file = os.path.join(recording, f"audio_array_{self.array_name}.wav")
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate

    def load_array_position(self, recording):
        # Load the groundtruth position of the array
        position_file = os.path.join(recording, f"position_array_{self.array_name}.txt")
        positions = pd.read_csv(position_file, delimiter="\t")
        return positions

    def load_source_waveform(self, recording):
        source_waveforms = {}
        for source in [path for path in os.listdir(recording) if ("audio_source" in path) and (".wav" in path)]:
            source_name = Path(source).stem.split("_")[-1]
            waveform, sample_rate = torchaudio.load(os.path.join(recording, source))
            source_waveforms[source_name] = {"waveform": waveform, "sample_rate": sample_rate}
        return source_waveforms

    def load_source_position(self, recording):
        source_positions = {}
        for source in [path for path in os.listdir(recording) if "position_source" in path]:
            source_name = Path(source).stem.split("_")[-1]
            positions = pd.read_csv(os.path.join(recording, source), delimiter="\t")
            source_positions[source_name] = positions
        return source_positions

    def load_array_vad(self, recording, source_names):
        # Load the VAD labels for the array
        vad_labels = {}
        for source_name in source_names:
            array_vad_file = os.path.join(recording, f"VAD_{self.array_name}_{source_name}.txt")
            array_vad_labels = torch.from_numpy(np.loadtxt(array_vad_file, skiprows=1))
            vad_labels[source_name] = array_vad_labels
        return vad_labels

    def load_source_vad(self, recording, source_names):
        # Load the VAD labels for the source
        vad_labels = {}
        for source_name in source_names:
            source_vad_file = os.path.join(recording, f"VAD_source_{source_name}.txt")
            source_vad_labels = torch.from_numpy(np.loadtxt(source_vad_file, skiprows=1))
            vad_labels[source_name] = source_vad_labels
        return vad_labels

    def load_source_timestamps(self, recording, sources):
        # Load the timestamps for the recording session
        timestamps = {}
        for source in sources:
            timestamps[source] = pd.read_csv(os.path.join(recording, f"audio_source_timestamps_{source}.txt"),
                                             delimiter="\t")
        return timestamps

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        # Get the recording folder name
        recording = self.recordings[idx]

        # Load array waveform
        waveform, sample_rate = self.load_array_waveform(recording)

        # Load array position data
        array_positions_all = self.load_array_position(recording)

        # Extract array information from position
        array_position = np.stack((array_positions_all['x'].values,
                                   array_positions_all['y'].values,
                                   array_positions_all['z'].values),
                                  axis=-1)
        array_reference_vector = np.stack(
            (array_positions_all['ref_vec_x'].values, array_positions_all['ref_vec_y'].values,
             array_positions_all['ref_vec_z'].values), axis=-1)
        array_rotation = np.zeros((array_position.shape[0], 3, 3))
        for i in range(3):
            for j in range(3):
                array_rotation[:, i, j] = array_positions_all['rotation_' + str(i + 1) + str(j + 1)]

        # Get array timestamps
        timestamp_array_file = os.path.join(recording, f"audio_array_timestamps_{self.array_name}.txt")
        timestamps_array = pd.read_csv(timestamp_array_file, delimiter="\t")
        array_timestamps = (timestamps_array['hour'].values * 3600
                            + timestamps_array['minute'].values * 60
                            + timestamps_array['second'].values)
        array_timestamps = array_timestamps - array_timestamps[0]

        # Get reference timestamps
        required_timestamps = pd.read_csv(os.path.join(recording, "required_time.txt"), delimiter="\t")
        required_timestamps = (required_timestamps['hour'].values * 3600
                               + required_timestamps['minute'].values * 60
                               + required_timestamps['second'].values)
        required_timestamps = required_timestamps - required_timestamps[0]

        if self.split_name == "dev":
            # Load source waveform
            source_waveforms = self.load_source_waveform(recording)

            # Load source position data
            source_positions_all = self.load_source_position(recording)

            # Load VAD labels (using a dummy source_name for now)
            source_information = source_waveforms.keys()
            array_vad_labels = self.load_array_vad(recording, source_names=source_information)
            source_vad_labels = self.load_source_vad(recording, source_names=source_information)

            # Load timestamps of source_information and array
            source_timestamps = self.load_source_timestamps(recording, source_information)

            # Compute trajectories
            source_information = {}
            for source_name, source_position in source_positions_all.items():
                position = np.stack((source_position["x"].values,
                                      source_position["y"].values,
                                      source_position["z"].values),
                                     axis=-1)
                relative_source_position = torch.from_numpy(
                    np.matmul(np.expand_dims(position - array_position, axis=1),
                              array_rotation).squeeze())  # np.matmul( array_rotation, np.expand_dims(sources_pos[s,...] - array_pos, axis=-1) ).squeeze()
                dist = torch.sqrt(torch.sum(torch.pow(relative_source_position, 2), dim=-1))
                doa_elevation = torch.rad2deg(torch.acos(relative_source_position[..., 2] / dist))
                doa_azimuth = torch.rad2deg(torch.atan2(relative_source_position[..., 1],
                                                        relative_source_position[..., 0]))
                source_information[source_name] = {"position": position,
                                                   "relative_position": relative_source_position,
                                                   "doa_elevation": doa_elevation,
                                                   "doa_azimuth": doa_azimuth,
                                                   "vad_labels": source_vad_labels[source_name]}

        else:
            array_vad_labels = None
            source_information = None

        if self.transform:
            waveform = self.transform(waveform)

        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'timestamps': array_timestamps,
            'array_position': array_position,
            'required_timestamps': required_timestamps,
            'array_vad_labels': array_vad_labels,
            'source_information': source_information
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = LOCATADataset(locata_root="/Users/JG96XG/Desktop/data_sets/LOCATA/", array_name="dummy", task="task5")
    a = dataset[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
