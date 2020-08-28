import json
from argparse import ArgumentParser
from pathlib import Path
from itertools import chain


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dstc_folder',
                        default='dstc8-schema-guided-dialogue/', type=Path)
    parser.add_argument('--output_folder', default='sgd/sgd/', type=Path)
    return parser.parse_args()


def main(args):
    for split in ['train', 'dev', 'test']:
        data = chain.from_iterable([
            json.loads(path.read_text())
            for path in args.dstc_folder.glob(f'{split}/dialogues_*.json')
        ])
        multiple = [d for d in data if len(d['services']) > 1]
        single = [d for d in data if len(d['services']) == 1]
        multiple_dir = args.output_folder / 'multiple' / split
        single_dir = args.output_folder / 'single' / split
        multiple_dir.mkdir(parents=True, exist_ok=True)
        single_dir.mkdir(parents=True, exist_ok=True)
        (multiple_dir / 'dialogues_001.json').write_text(json.dumps(multiple))
        (single_dir / 'dialogues_001.json').write_text(json.dumps(single))


if __name__ == '__main__':
    main(parse_args())
