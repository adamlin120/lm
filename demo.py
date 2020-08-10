import logging
from argparse import ArgumentParser, Namespace
from typing import List
from itertools import cycle

import torch
from pytorch_lightning import Trainer
from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer

from module import ConditionalLM, build_input_from_segments, SYSTEM

logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = GPT2DoubleHeadsModel.from_pretrained(args.checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint_path)
    if torch.cuda.is_available():
        model.cuda()

    eos_token_id = tokenizer.convert_tokens_to_ids('<eos>')

    def prepare(history: List[str], cue: str):
        instance = build_input_from_segments(
            history=history,
            next_user_utterance=cue,
            reply='',
            is_ground_truth_reply=False,
            tokenizer=tokenizer
        )
        instance.pop('mc_token_ids')
        instance.pop('labels')
        instance['input_ids'].pop(-1)
        instance['token_type_ids'].pop(-1)
        return {k: torch.tensor([v], device=model.device)
                for k, v in instance.items()}

    while True:
        print('\nStarting a new dialog. '
              'To end a dialog enter /stop. '
              'Interrupt to end this program.\n')
        history = []
        while True:
            if history:
                print('---Dialog Start---')
                for speaker, utt in zip(cycle(['USER', 'SYSTEM']), history):
                    print(f"{speaker}: {utt}")
            user_utternce = input('USER: ')
            if user_utternce == '/stop':
                break
            history.append(user_utternce)
            cue = input('Cues (expected user reply): ')
            tensors = prepare(history, cue)
            preds = model.generate(pad_token_id=50256,
                                   eos_token_id=eos_token_id,
                                   max_length=512,
                                   **tensors)
            preds_txt = [tokenizer.decode(pred) for pred in preds]
            pred_txt = preds_txt[0]
            sys_reply_start = pred_txt.rfind(SYSTEM)
            sys_reply = ' '.join(pred_txt[sys_reply_start:].split()[1:-1])
            history.append(sys_reply)


def parse_args() -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('checkpoint_path')
    parser = ConditionalLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
