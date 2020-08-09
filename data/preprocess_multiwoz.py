import json
from pathlib import Path
from collections import defaultdict
from itertools import chain


def main():
    datasets = defaultdict(dict)
    dir = Path('./multiwoz2.1_nadst/')
    paths = {
        'train': dir / 'nadst_train_dials.json',
        'valid': dir / 'nadst_dev_dials.json',
        'test': dir / 'nadst_test_dials.json',
    }
    for split, path in paths.items():
        for dial in json.loads(path.read_text()):
            dial_idx = dial['dialogue_idx']
            turns = list(chain.from_iterable(
                [turn["delex_system_transcript"], turn["delex_transcript"]]
                for turn in dial['dialogue']
            ))
            del turns[0]
            assert len(turns) % 2
            datasets[split][dial_idx] = turns
    assert all(split in datasets for split in ['train', 'valid'])
    Path('multiwoz2.1.processed.json').write_text(json.dumps(datasets,
                                                             indent=2))


if __name__ == '__main__':
    main()
