import os

import numpy as np
import pyroomacoustics as pra
import random
import torch
import torchaudio
from torchaudio.functional import resample

from torch_audiomentations import AddBackgroundNoise
import pyfar as pf

from source.utils import spherical_to_cartesian


class RoomSimulator:
    """
    RoomSimulator is a class to simulate audio from a source position in a ShoeBox room.
    """

    def __init__(self, room_dim, rt60, use_rand_ism=True, max_rand_disp=0.05,
                 sampling_rate=16000):
        self.room_dim = room_dim
        self.sampling_rate = sampling_rate

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        self.room = pra.ShoeBox(room_dim, fs=sampling_rate, max_order=max_order, materials=pra.Material(e_absorption),
                                use_rand_ism=use_rand_ism, max_rand_disp=max_rand_disp)

    def load_audio(self, filepath):
        waveform, fs = torchaudio.load(filepath)
        if fs != self.sampling_rate:
            waveform = resample(waveform, fs, sampling_rate)
        return waveform.numpy().squeeze(), fs

    def add_source(self, source_position, signal):
        self.room.add_source(np.array(source_position), signal=signal)

    def add_microphone(self, microphone_position):
        self.room.add_microphone_array(pra.MicrophoneArray(np.expand_dims(np.array(microphone_position), -1),
                                                           fs=self.sampling_rate))

    def plot_room(self):
        self.room.plot()

    def compute_rir(self):
        # Compute the RIR using the hybrid method
        self.room.compute_rir()

    def simulate_room_acoustics(self):
        """
        Simulates the room acoustics.
        Returns the room's output signal.
        """
        self.room.simulate()
        return self.room.mic_array.signals


class BinauralAudioSimulator:
    """
    Generates binaural audio using HRTF from .sofa files.
    """

    def __init__(self, hrtf_folder, sampling_rate=16000):
        self.hrtf_folder = hrtf_folder
        self.sampling_rate = sampling_rate
        # Precompute the list of available HRTF files in the specified folder
        self.available_hrtf_files = self._list_hrtf_files(hrtf_folder)
        self.sampling_rate = sampling_rate

    @staticmethod
    def _list_hrtf_files(hrtf_folder):
        if isinstance(hrtf_folder, str):
            hrtf_folder = [hrtf_folder]
        hrtf_files = []
        for folder in hrtf_folder:
            for root, _, files in os.walk(folder):
                files = [os.path.join(root, file) for file in files if file.endswith('.sofa')]
                hrtf_files.extend(files)
        return hrtf_files

    def _select_random_hrtf(self, source_direction):
        selected_hrtf_file = random.choice(self.available_hrtf_files)
        data_ir, source_coordinates, receiver_coordinates = pf.io.read_sofa(selected_hrtf_file, verbose=False)
        index, distance = source_coordinates.find_nearest(pf.Coordinates(*source_direction, domain="cart"),
                                                          k=1,
                                                          distance_measure='euclidean')
        hrtf_left = data_ir.time[index[0], 0]
        hrtf_right = data_ir.time[index[0], 1]
        sampling_rate = data_ir.sampling_rate
        return hrtf_left, hrtf_right, sampling_rate, receiver_coordinates.cartesian

    def load_speech(self, filepath):
        waveform, fs = torchaudio.load(filepath)
        if fs != self.sampling_rate:
            waveform = resample(waveform, fs, self.sampling_rate)
        return waveform.numpy().squeeze(), fs

    def apply_hrtf(self, signal, source_direction):
        """
        Applies the HRTF to the given signal to create a binaural audio signal.
        """
        hrtf_left, hrtf_right, sampling_rate, receiver_coordinates = self._select_random_hrtf(source_direction)
        if sampling_rate != self.sampling_rate:
            signal = resample(signal, self.sampling_rate, sampling_rate).float()

        hrtf_left = torch.from_numpy(hrtf_left).float().view(1, 1, -1)
        hrtf_right = torch.from_numpy(hrtf_right).float().view(1, 1, -1)

        normalization_factor = torch.max(torch.max(torch.abs(hrtf_left)), torch.max(torch.abs(hrtf_right)))

        hrtf_left = hrtf_left / (normalization_factor + 1.e-9)
        hrtf_right = hrtf_right / (normalization_factor + 1.e-9)

        right = torchaudio.functional.fftconvolve(signal, hrtf_right, mode='full')
        left = torchaudio.functional.fftconvolve(signal, hrtf_left, mode='full')

        if sampling_rate != self.sampling_rate:
            left = resample(left, sampling_rate, self.sampling_rate).float()
            right = resample(right, sampling_rate, self.sampling_rate).float()
        if torch.isnan(left).any():
            print("clean input Tensor contains NaN values")
        if torch.isnan(right).any():
            print("clean input Tensor contains NaN values")
        return left, right, receiver_coordinates

    def __call__(self, signal, source_direction):
        return self.apply_hrtf(signal, source_direction)


if __name__ == '__main__':
    import sounddevice as sd
    import time
    from matplotlib import pyplot as plt

    # Simulation parameters
    sampling_rate = 16000

    room_dim = (15., 10., 3.)
    rt60 = 0.3
    room_simulator = RoomSimulator(room_dim=room_dim, rt60=rt60)

    microphone_position = (7., 5., 1.5)
    room_simulator.add_microphone(microphone_position=microphone_position)
    relative_source_position = (-5., 0., 0.)
    source_position = tuple(m + s for m, s in zip(microphone_position, relative_source_position))

    noise_files_path = "/Users/JG96XG/Desktop/data_sets/noise_files/kolbaek_noise_files/bus/"

    # hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/RIEC_hrtf_all/"
    # hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/FABIAN_HRTF_DATABASE_v4/1 HRIRs/SOFA/"
    #hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/Database-Master_V1-4/D1/D1_BRIR_SOFA"
    # hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/Aachen_ITA HRTF/HRTF-Database/SOFA"
    # hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/ARI/ari/"
    # hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/CIPIC_sofa/"
    hrtf_folder = "/Users/JG96XG/Desktop/data_sets/HRTFs/ARI/ari_bte_splits/"

    # 1. Load the speech signal
    speech_signal, fs = torchaudio.load(
        '/Users/JG96XG/PycharmProjects/aisnet/data/OSR_us_000_0010_8k.wav')

    if fs != sampling_rate:
        speech_signal = resample(speech_signal, fs, sampling_rate)

    duration = 5.0
    # sd.play(speech_signal[0], sampling_rate)
    # time.sleep(duration)
    # sd.stop()

    speech_signal = speech_signal.numpy()[0]
    # 2. Add the source to the room and add listener position
    room_simulator.add_source(source_position=source_position, signal=speech_signal)
    # room_simulator.plot_room()
    # plt.show()

    # 3. Simulate room
    room_simulator.compute_rir()
    room_output = room_simulator.simulate_room_acoustics()

    duration = 5.0
    # sd.play(room_output[0], sampling_rate)
    # time.sleep(duration)
    # sd.stop()

    # 4. Add background noise
    noise_augmenter = AddBackgroundNoise(background_paths=noise_files_path, p=1.0,
                                         min_snr_in_db=10, max_snr_in_db=10, sample_rate=sampling_rate)
    noisy_room_output = noise_augmenter(torch.from_numpy(room_output).unsqueeze(0))

    sd.play(noisy_room_output[0].T, sampling_rate)
    time.sleep(duration)
    sd.stop()

    # 5. Apply the HRTF to generate the final binaural audio
    head_direction = (1., -90, 45)  # Right, horizontal plane
    head_direction = spherical_to_cartesian(*head_direction)

    hrtf_simulator = BinauralAudioSimulator(hrtf_folder=hrtf_folder,
                                            sampling_rate=sampling_rate)
    left_signal, right_signal, receiver_position = hrtf_simulator(noisy_room_output.float(),
                                                                  source_direction=head_direction)

    stereo = torch.cat((left_signal, right_signal), dim=1)
    #stereo = stereo / (stereo.max() + 1.e-9)
    duration = 5.0
    sd.play(stereo[0].T, sampling_rate)
    time.sleep(duration)
    sd.stop()

    # 6. Save the output binaural audio
    torchaudio.save('output_binaural.flac', stereo[0], sampling_rate)

    specgram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=400)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(specgram(stereo)[0, 0])
    ax[1].imshow(specgram(stereo)[0, 1])
    plt.plot()

    print("done")
