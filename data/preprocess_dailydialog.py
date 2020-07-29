import json
from pathlib import Path


SPLITS = ['train', 'valid', 'test']
SENT_DELIMITER = "__eou__"


def main():
    dir = Path('./dailydialog/')
    output_path = Path(f"dailydialog.processed.json")
    output_debug_path = Path(f"dailydialog.processed.debug.json")

    datasets = {}
    for split in ['train', 'validation', 'test']:
        path = dir / split / f'dialogues_{split}.txt'
        datasets[split.replace('validation', 'valid')] = \
            {
                i: line.strip().split(SENT_DELIMITER)
                for i, line in enumerate(path.read_text().splitlines())
            }
    assert all(split in datasets for split in SPLITS)
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
