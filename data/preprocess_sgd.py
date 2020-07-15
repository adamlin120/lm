import json
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import cycle
from argparse import ArgumentParser


def main(delex: bool = False):
    dir = Path('./dstc8-schema-guided-dialogue/')
    output_path = Path(f"sgd.processed{'.delex' if delex else ''}.json")
    output_debug_path = Path(
        f"sgd.processed{'.delex' if delex else ''}.debug.json")

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
                    if delex:
                        utterance = delexicalize(utterance, turn['frames'])
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
    output_debug_path.write_text(json.dumps(debug_dataset,
                                                           indent=2))


def delexicalize(utterance: str, frames: List[Dict]) -> str:
    mentions: List[Tuple[str, str]] = []
    for frame in frames:
        slots = frame.get('slots', [])
        for slot in slots:
            start = slot['start']
            end = slot['exclusive_end']
            slot_name = slot['slot']
            slot_value = utterance[start:end]
            mentions.append((slot_name, slot_value))

    for name, value in mentions:
        utterance = utterance.replace(value, name)
        # TODO:
        assert value not in utterance and name in utterance, \
            f"Utterance: {utterance}\nValue: {slot_value}\nName: {name}"
    return utterance


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--delex', action='store_true')
    args = parser.parse_args()
    main(args.delex)
