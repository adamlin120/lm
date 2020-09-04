import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from module import ConditionalLM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = ConditionalLM.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hparams_file=args.hparams_file,
    )
    args.save_dir.mkdir(exist_ok=True, parents=True)
    model.model.save_pretrained(args.save_dir)
    model.tokenizer.save_pretrained(args.save_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("checkpoint_path")
    parser.add_argument("hparams_file")
    parser.add_argument("save_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
