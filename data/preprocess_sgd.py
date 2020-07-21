import json
from pathlib import Path
from typing import List
from itertools import cycle


def main():
    dir = Path('./dstc8-schema-guided-dialogue/')
    output_path = Path(f"sgd.processed.json")
    output_debug_path = Path(f"sgd.processed.debug.json")

    split_dirs = {
        'train': dir / 'train',
        'valid': dir / 'dev',
        'test': dir / 'test',
    }
    datasets = {k: {} for k in split_dirs.keys()}
    for split, dir in split_dirs.items():
        for path in dir.glob('dialogues_*.json'):
            for dial in json.loads(path.read_text()):
                dial_idx = dial['dialogue_id']
                turns: List[str] = []
                for speaker, turn in zip(cycle(('USER', 'SYSTEM')),
                                         dial['turns']):
                    assert speaker == turn['speaker']
                    utterance = turn['utterance']
                    turns.append(utterance)
                datasets[split][dial_idx] = turns
    assert all(split in datasets for split in split_dirs.keys())
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
