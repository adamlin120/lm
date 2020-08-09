import json
from pathlib import Path


DIAL_START = "[STARTCONVERSATION]"


def main():
    path = Path('./sgd/delex/')
    output_path = Path(f"sgd.processed.delex.json")
    output_debug_path = Path(f"sgd.processed.delex.debug.json")

    split_files = {
        'train': path / 'train.txt',
        'valid': path / 'dev.txt',
        'test': path / 'test.txt',
    }
    datasets = {
        split: {i: dial.strip().splitlines()
                for i, dial in enumerate(path.read_text().split(DIAL_START))
                if dial.strip()
                }
        for split, path in split_files.items()
    }
    assert all(split in datasets for split in split_files.keys())
    output_path.write_text(json.dumps(datasets, indent=2))

    debug_dataset = {
        split: {
            dial_idx: turns
            for dial_idx, turns in list(dataset.items())[:10]
        }
        for split, dataset in datasets.items()
    }
    output_debug_path.write_text(json.dumps(debug_dataset, indent=2))


if __name__ == '__main__':
    main()
