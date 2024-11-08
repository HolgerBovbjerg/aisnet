import os
import random
import shutil
from pathlib import Path
import argparse

from numpy import allclose


def split_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None, overwrite=False):
    assert allclose(train_ratio + val_ratio + test_ratio, 1.0), "train_ratio, val_ratio and test_ratio must sum to 1.0"

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Ensure the output directories exist or handle existing directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"

    for path in [train_dir, val_dir, test_dir]:
        if path.exists():
            if not overwrite:
                print(f"Directory {path} already exists. Use '--overwrite' to overwrite.")
                return
            else:
                print(f"Overwriting existing directory: {path}")
                shutil.rmtree(path)  # Delete existing directory if overwriting
        path.mkdir(parents=True, exist_ok=True)

    # Get all .sofa files in the input directory
    sofa_files = [f for f in os.listdir(input_dir) if f.endswith('.sofa')]
    total_files = len(sofa_files)

    # Shuffle the files
    random.shuffle(sofa_files)

    # Determine split sizes
    train_size = int(train_ratio * total_files)
    val_size = int(val_ratio * total_files)

    # Split files
    train_files = sofa_files[:train_size]
    val_files = sofa_files[train_size:train_size + val_size]
    test_files = sofa_files[train_size + val_size:]

    # Copy files to respective folders
    for file in train_files:
        shutil.copy(Path(input_dir) / file, train_dir / file)
    for file in val_files:
        shutil.copy(Path(input_dir) / file, val_dir / file)
    for file in test_files:
        shutil.copy(Path(input_dir) / file, test_dir / file)

    print(f"Total files: {total_files}")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Split a folder of .sofa files into train, val, and test sets.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing .sofa files.")
    parser.add_argument("--output_dir", type=str,
                        help="Path to the output directory where the split files will be stored.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (optional).",
                        required=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directories if they exist.")

    args = parser.parse_args()

    # Run the split function with the provided arguments
    split_data(args.input_dir, args.output_dir, seed=args.seed, overwrite=args.overwrite)
