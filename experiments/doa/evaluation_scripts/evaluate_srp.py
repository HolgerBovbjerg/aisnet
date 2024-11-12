import numpy as np
import torch

from source.datasets.binaural_librispeech.BinauralLibriSpeechDataset import BinauralLibriSpeechDataset
from source.models.doa import SRPPHATDoA
from source.metrics import angular_error, doa_threshold_accuracy


if __name__ == '__main__':
    def spherical_to_cartesian(r, theta, phi):
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    root_dir = "/Users/JG96XG/Desktop/data_sets/BinauralLibriSpeech/horizontal_plane_front_only"
    metadata_filename = "metadata.csv"
    split = "test-other"
    dataset = BinauralLibriSpeechDataset(root_dir=root_dir, metadata_filename=metadata_filename, split=split)

    elevation_range = (90, 90)
    azimuth_range = (-90, 90)
    frequency_range = (500., 4000.)

    angular_errors = []
    absolute_errors = []

    for data in dataset:
        microphone_positions = data["microphone_positions"]
        srp_phat = SRPPHATDoA(n_fft=512, window_length=400, hop_length=160,
                              sample_rate=16000,
                              microphone_positions=microphone_positions, center=True,
                              elevation_range=elevation_range, azimuth_range=azimuth_range,
                              frequency_range=frequency_range, elevation_resolution=10., azimuth_resolution=5.)
        waveform = data["waveform"]
        # duration = 5.0
        # sd.play(waveform.T, dataset.sample_rate)
        # time.sleep(duration)
        # sd.stop()
        doa_true = (data["elevation"], data["azimuth"])
        doa = srp_phat(waveform.unsqueeze(0))
        time_steps = doa.size(1)
        elevations_true = (doa_true[0]*torch.ones(time_steps))
        elevations = (doa[..., 0].squeeze())
        azimuths_true = (doa_true[1]*torch.ones(time_steps))
        azimuths = (doa[..., 1].squeeze())
        doa_pred = spherical_to_cartesian(torch.ones(time_steps),
                                          torch.deg2rad(elevations),
                                          torch.deg2rad(azimuths))
        doa_true = spherical_to_cartesian(torch.ones(time_steps),
                                          torch.deg2rad(elevations_true),
                                          torch.deg2rad(azimuths_true))
        errors = angular_error(doa_pred, doa_true)
        angular_errors.append(errors)
        absolute_errors.append(torch.abs(azimuths_true - azimuths))

    mean_angular_error = torch.rad2deg(torch.mean(torch.cat(angular_errors)))
    threshold_accuracy = torch.rad2deg(doa_threshold_accuracy(torch.cat(angular_errors),
                                                              threshold_radians=np.deg2rad(10.)))
    print("done")
