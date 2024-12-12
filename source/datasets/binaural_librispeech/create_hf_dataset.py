import os
from pathlib import Path

from datasets import DatasetDict, Audio
import pandas as pd
from datasets.table import embed_table_storage
import argparse
from datasets import load_from_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("main_folder_path", type=str, help="Path of the base librispeech_binaural folder")
    parser.add_argument("subset", type=str,
                        help="Dataset subset to use, if necessary.")
    parser.add_argument("output_dir", type=str, help="Save the dataset on disk with this path.")

    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers.")
    parser.add_argument("--csv_folder_path", default=None, type=str,
                        help="Path where to save intermediate csv, by default will be main_folder_path")
    parser.add_argument("--repo_id", default="Holger1997/BinauralLibriSpeech", type=str,
                        help="Push the dataset to the hub.")
    parser.add_argument("--ref", default="None", type=str,
                        help="Push reference branch.")

    args = parser.parse_args()

    try:
        dataset = load_from_disk(os.path.join(args.output_dir, args.subset))
    except FileNotFoundError:
        main_folder_path = args.main_folder_path
        csv_folder_path = args.csv_folder_path if args.csv_folder_path is not None \
            else os.path.join(main_folder_path, args.subset)
        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)

        splits = ['train-clean-100',
                  #'train-clean-360',
                  #'train-other-500',
                  'dev-clean',
                  'dev-other',
                  'test-clean',
                  'test-other']

        csv_dict = {}
        for split in splits:
            metadata_path = os.path.join(main_folder_path, args.subset, split, "metadata.csv")
            df = pd.read_csv(str(metadata_path), index_col="file_name")

            split = split.replace("-", "_")
            df["split"] = split

            print(f"len df {len(df)}")

            df.to_csv(os.path.join(csv_folder_path, f"{split}.csv"))
            csv_dict[split] = os.path.join(csv_folder_path, f"{split}.csv")

        dataset = DatasetDict.from_csv(csv_dict)


        def extract_speaker_and_chapter_id(file_name, split):
            file_name = Path(file_name).stem
            speaker_id = file_name.split("-")[0]
            chapter_id = file_name.split("-")[1]
            file = file_name + ".flac"
            path = os.path.join(main_folder_path, args.subset, split.replace("_", "-"), speaker_id, chapter_id, file)
            return {"audio": path, "speaker_id": speaker_id, "chapter_id": chapter_id, "file": file}


        # correct audio path
        dataset = dataset.map(extract_speaker_and_chapter_id, input_columns=["file_name", "split"],
                              num_proc=args.cpu_num_workers, remove_columns=["file_name", "split"])
        dataset = dataset.cast_column("audio", Audio(mono=False))

        def extract_durations(audio):
            return {"duration": audio["array"].shape[-1] / audio["sampling_rate"]}

        dataset = dataset.map(extract_durations, input_columns=["audio"], num_proc=args.cpu_num_workers)

        print(dataset)
        print(dataset["test_clean"][0])

        print("Embed table storage")

        # Get dataset format
        dataset_format = dataset["test_clean"].format
        # Change format to arrow
        dataset = dataset.with_format("arrow")
        # Embed external data into a Pyarrow table's storage
        dataset = dataset.map(embed_table_storage, batched=True, num_proc=args.cpu_num_workers)
        # Change back format
        dataset = dataset.with_format(**dataset_format)

        dataset.save_to_disk(os.path.join(args.output_dir, args.subset), num_proc=args.cpu_num_workers)

    if args.repo_id:
        pushed = False
        while not pushed:
            try:
                dataset.push_to_hub(args.repo_id, args.subset, revision=args.ref)
                pushed = True
            except:
                pass
