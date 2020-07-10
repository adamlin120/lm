import logging
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from module import ConditionalLM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    trainer = Trainer.from_argparse_args(args)
    model = ConditionalLM(args)
    trainer.fit(model)


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=13)

    parser = ConditionalLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(accumulate_grad_batches=4,
                        gradient_clip_val=1.0,
                        )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
