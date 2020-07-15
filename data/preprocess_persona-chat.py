import json
from pathlib import Path


SPLITS = ['train', 'valid', 'test']


def main():
    dir = Path('./persona-chat/personachat_self_original.json')
    output_path = Path(f"persona.processed.json")
    output_debug_path = Path(f"persona.processed.debug.json")

    raw_data = json.loads(dir.read_text())
    datasets = {k: {} for k in SPLITS}
    for split, dials in raw_data.items():
        num_instance = len(dials)
        for i, dial in enumerate(dials):
            if split == 'valid' and i > num_instance // 2:
                split = 'test'
            dial_idx = f'{split}_{i}'
            turns = dial['utterances'][-1]['history']
            turns.append(dial['utterances'][-1]['candidates'][-1])
            datasets[split][dial_idx] = turns
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
