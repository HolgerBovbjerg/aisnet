# Load model directly
from transformers import Wav2Vec2Model, AutoProcessor
import torch

from source.utils import count_parameters
from source.datasets.binaural_librispeech import BinauralLibriSpeechDataset


processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

parameters = count_parameters(model)

root_dir = "/Users/JG96XG/Desktop/data_sets/BinauralLibriSpeech/horizontal_plane_front_only"
metadata_filename = "metadata.csv"
split = "test-other"
dataset = BinauralLibriSpeechDataset(root_dir=root_dir, metadata_filename=metadata_filename, split=split)

for data in dataset:
    microphone_positions = data["microphone_positions"]
    waveform = data["waveform"]
    left = waveform[0]
    right = waveform[1]
    with torch.no_grad():
        left_out = model(left.view(1, -1))["extract_features"]
        right_out = model(right.view(1, -1))["extract_features"]

    out = torch.cat((left_out, right_out), dim=-1)

print("done")