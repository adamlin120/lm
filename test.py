import logging
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer

from module import ConditionalLM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = ConditionalLM.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hparams_file=args.hparams_file,
    )
    if args.interactive:
        prompt = input("Input: ")
        while prompt.strip():
            print(model.interactive(prompt))
    else:
        trainer = Trainer.from_argparse_args(args)
        trainer.test(model)


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('checkpoint_path')
    parser.add_argument('hparams_file')
    parser.add_argument('--interactive', action='store_true')
    parser = ConditionalLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
