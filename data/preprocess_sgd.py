import json
from pathlib import Path
from argparse import ArgumentParser


DIAL_START = "[STARTCONVERSATION]"


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("version", type=str)
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path = Path(f"sgd.{args.version}.processed.delex.json")
    output_debug_path = Path(f"sgd.{args.version}.processed.delex.debug.json")

    split_files = {
        "train": args.data_path / "train.txt",
        "valid": args.data_path / "dev.txt",
        "test": args.data_path / "test.txt",
    }
    datasets = {
        split: {
            i: dial.strip().splitlines()
            for i, dial in enumerate(path.read_text().split(DIAL_START))
            if dial.strip()
        }
        for split, path in split_files.items()
    }
    assert all(split in datasets for split in split_files.keys())
    output_path.write_text(json.dumps(datasets, indent=2))

    debug_dataset = {
        split: {dial_idx: turns for dial_idx, turns in list(dataset.items())[:10]}
        for split, dataset in datasets.items()
    }
    output_debug_path.write_text(json.dumps(debug_dataset, indent=2))


if __name__ == "__main__":
    main()
