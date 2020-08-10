import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from module import ConditionalLM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = ConditionalLM.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hparams_file=args.hparams_file,
    )
    trainer = Trainer.from_argparse_args(args)
    dataset = torch.load(args.tensor_dataset_cache.open('rb'))
    test_dataloaders = DataLoader(
        dataset,
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_workers,
        pin_memory=True
    )
    trainer.test(model, test_dataloaders)


def parse_args() -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('checkpoint_path')
    parser.add_argument('hparams_file')
    parser.add_argument('--tensor_dataset_cache', type=Path)
    parser = ConditionalLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
