import numpy as np
import torch

from source.datasets.binaural_librispeech.BinauralLibriSpeechDataset import BinauralLibriSpeechDataset
from source.models.doa import GCCDoA
from source.metrics import angular_error, threshold_accuracy


if __name__ == '__main__':
    def spherical_to_cartesian(r, theta, phi):
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    root_dir = "/Users/JG96XG/Desktop/data_sets/BinauralLibriSpeech/horizontal_plane_front_only"
    metadata_filename = "metadata.csv"
    split = "dev-clean"
    dataset = BinauralLibriSpeechDataset(root_dir=root_dir, metadata_filename=metadata_filename, split=split)

    elevation_range = (90, 90)
    azimuth_range = (-90, 90)

    angular_errors = []
    absolute_errors = []

    for i in range(len(dataset)):
        data = dataset[i]
        microphone_positions = data["microphone_positions"]
        gcc_phat = GCCDoA(n_fft=512, window_length=400, hop_length=160, sample_rate=dataset.sample_rate,
                              microphone_positions=microphone_positions, center=True, limit_to_max_delay=True)
        waveform = data["waveform"]
        # duration = 5.0
        # sd.play(waveform.T, dataset.sample_rate)
        # time.sleep(duration)
        # sd.stop()
        doa_true = (data["elevation"], data["azimuth"])
        doa = gcc_phat(waveform.unsqueeze(0))

        time_steps = doa.size(0)
        elevations_true = doa_true[0] * torch.ones(time_steps)
        azimuths_true = doa_true[1] * torch.ones(time_steps)
        azimuths = doa
        doa_pred = spherical_to_cartesian(torch.ones(time_steps),
                                          torch.deg2rad(elevations_true),
                                          torch.deg2rad(azimuths))
        doa_true = spherical_to_cartesian(torch.ones(time_steps),
                                          torch.deg2rad(elevations_true),
                                          torch.deg2rad(azimuths_true))
        errors = angular_error(doa_pred, doa_true)
        angular_errors.append(errors)
        absolute_errors.append(torch.abs(azimuths_true - azimuths))

    mean_angular_error = torch.rad2deg(torch.mean(torch.cat(angular_errors)))
    threshold_accuracy = torch.rad2deg(threshold_accuracy(torch.cat(angular_errors),
                                       threshold=np.deg2rad(10.)))
    print("done")
