import json
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output_path', type=Path)
    parser.add_argument('datasets', nargs='+', type=Path)
    args = parser.parse_args()

    datasets = [json.loads(path.read_text()) for path in args.datasets]
    combined_dataset = defaultdict(dict)
    for dataset in datasets:
        for k, v in dataset.items():
            combined_dataset[k] = {**v, **combined_dataset[k]}
    args.output_path.write_text(json.dumps(combined_dataset, indent=2))
