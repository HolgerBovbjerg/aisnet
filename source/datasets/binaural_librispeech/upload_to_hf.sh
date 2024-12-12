#!/usr/bin/env bash
SCRIPT_DIR=$(dirname "$0")
python "${SCRIPT_DIR}/create_hf_dataset.py" /Users/JG96XG/Desktop/data_sets/BinauralLibriSpeech horizontal_plane_front_only /Users/JG96XG/Desktop/data_sets/HF/BinauralLibriSpeech --cpu_num_workers 8 --ref "refs/pr/6"