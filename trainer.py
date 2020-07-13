import logging
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from module import ConditionalLM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    tb_logger = loggers.TensorBoardLogger('logs/')
    wandb_logger = loggers.WandbLogger(save_dir='logs/',
                                       project="controllable-response"
                                       )
    checkpoint_callback = ModelCheckpoint(
        filepath='ckpts/{epoch}-{val_loss:.4f}-{val_bleu:.2f}',
        save_last=True,
        save_top_k=2,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    trainer = Trainer.from_argparse_args(args,
                                         logger=[tb_logger, wandb_logger],
                                         checkpoint_callback=checkpoint_callback
                                         )
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
