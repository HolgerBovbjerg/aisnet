import random
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm
from torchaudio.datasets import LIBRISPEECH
from torchaudio.functional import resample
import pandas as pd
import torch
import torchaudio
from torch_audiomentations import ApplyImpulseResponse

from source.simulators import BinauralAudioSimulator
from source.utils import spherical_to_cartesian


def process(data, sampling_rate, elevation_resolution, elevation_range, azimuth_resolution, azimuth_range, rir_simulator, hrtf_simulator, save_root):
    waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = data
    if sample_rate != sampling_rate:
        waveform = resample(waveform, sample_rate, sampling_rate)

    waveform = rir_simulator(waveform.unsqueeze(0))[0]
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1.e-9)

    elevation_span = abs(elevation_range[1] - elevation_range[0])
    azimuth_span = abs(azimuth_range[1] - azimuth_range[0])
    if elevation_span:
        elevation = random.randrange(elevation_range[0], elevation_range[1], elevation_resolution)
    else:
        elevation = elevation_range[0]
    if azimuth_span:
        azimuth = random.randrange(azimuth_range[0], azimuth_range[1], azimuth_resolution)
    else:
        azimuth = azimuth_range[0]

    source_direction = (1., elevation, azimuth)  # Left, horizontal plane
    source_direction_cart = spherical_to_cartesian(*source_direction)
    left_signal, right_signal, receiver_position = hrtf_simulator(waveform,
                                                                  source_direction=source_direction_cart)
    stereo = torch.cat((left_signal, right_signal), dim=0).squeeze()
    stereo = stereo / (torch.max(torch.abs(stereo)) + 1.e-9)

    save_folder = f"{speaker_id}/{chapter_id}"
    if not Path(f"{save_root}/{save_folder}/").exists():
        Path(f"{save_root}/{save_folder}/").mkdir(parents=True, exist_ok=True)
    save_name = "-".join([str(speaker_id), str(chapter_id), f"{utterance_id:04d}"])
    save_path = f"{save_root}/{save_folder}/{save_name}"

    torchaudio.save(uri=save_path + ".flac", src=stereo, sample_rate=sampling_rate)
    metadata = {"file_name": f"{save_folder}/{save_name}.flac",
                "elevation": elevation,
                "azimuth": azimuth,
                "microphone_left_x": receiver_position[0, 0, 0],
                "microphone_left_y": receiver_position[0, 0, 1],
                "microphone_left_z": receiver_position[0, 0, 2],
                "microphone_right_x": receiver_position[1, 0, 0],
                "microphone_right_y": receiver_position[1, 0, 1],
                "microphone_right_z": receiver_position[1, 0, 2],
                "transcript": transcript}
    return metadata


if __name__ == "__main__":
    # LibriSpeech dataset
    splits = ["train-clean-100"]  # ["dev-clean", "test-clean", "dev-other", "test-other"]
    root = "/Users/JG96XG/Desktop/data_sets/"

    sampling_rate = 16000

    add_rir = True
    horizontal = True
    front_only = True
    seed = 42
    random.seed(seed)
    n_processes = 8
    elevation_resolution = 10
    azimuth_resolution = 5

    # create RIR simulator
    ir_paths = "/Users/JG96XG/Desktop/data_sets/rirs_noises/simulated_rirs/"
    rir_simulator = ApplyImpulseResponse(ir_paths=ir_paths,
                                         sample_rate=sampling_rate,
                                         p=1.0) if add_rir else lambda x: x

    for split in splits:
        dataset = LIBRISPEECH(root=root, url=split, folder_in_archive='LibriSpeech', download=True)
        # Create hrtf simulator using RIEC HRTFs
        if split in ('train-clean-100', 'train-clean-360', 'train-other-500'):
            hrtf_split = "train"
        elif split in ('dev-clean', 'dev-other'):
            hrtf_split = "val"
        elif split in ('test-clean', 'test-other'):
            hrtf_split = "test"
        else:
            raise ValueError(f"Split {split} not recognized")

        hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/ARI/ari_splits/" + hrtf_split
        hrtf_simulator = BinauralAudioSimulator(hrtf_folder=hrtf_folder, sampling_rate=sampling_rate)

        if horizontal:
            elevation_range = (90, 90)
            if front_only:
                azimuth_range = (-90, 90)
                save_root = root + f"BinauralLibriSpeech/horizontal_plane_front_only/{split}"
            else:
                azimuth_range = (-180, 180)
                save_root = root + f"BinauralLibriSpeech/horizontal_plane/{split}"
        else:
            elevation_range = (40, 140)
            if front_only:
                azimuth_range = (-90, 90)
                save_root = root + f"BinauralLibriSpeech/spherical_front_only/{split}"
            else:
                azimuth_range = (-180, 180)
                save_root = root + f"BinauralLibriSpeech/spherical/{split}"
                elevation_range = (40, 140)

        func = partial(process, sampling_rate=sampling_rate, elevation_resolution=elevation_resolution,
                       elevation_range=elevation_range, azimuth_resolution=azimuth_resolution,
                       azimuth_range=azimuth_range, rir_simulator=rir_simulator,
                       hrtf_simulator=hrtf_simulator, save_root=save_root)

        metadata = []
        pool = Pool(processes=n_processes)
        for result in tqdm(pool.imap(func=func, iterable=dataset), total=len(dataset)):
            metadata.append(result)

        pool.close()
        pool.join()
        metadata = pd.DataFrame(metadata)
        metadata.to_csv(f"{save_root}/metadata.csv", index=False)

    print("done")
