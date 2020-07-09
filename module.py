import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, \
    GPT2DoubleHeadsModel, GPT2Tokenizer

from utils import get_dataset

IGNORE_INDEX = -100
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
SPEAKER1 = "<speaker1>"
SPEAKER2 = "<speaker2>"
SPECIAL_TOKENS = [BOS, EOS, SPEAKER1, SPEAKER2, PAD]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': BOS, 'eos_token': EOS,
                         'pad_token': PAD,
                         'additional_special_tokens': [SPEAKER1, SPEAKER2]}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "labels", "mc_labels",
                "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]


class ConditionalLM(LightningModule):
    def __init__(self,
                 hparams: Namespace,
                 ):
        super().__init__()
        self.hparams = hparams

        is_gpt2 = "gpt2" in self.hparams.model_checkpoint
        tokenizer_class = GPT2Tokenizer if is_gpt2 else OpenAIGPTTokenizer
        model_class = GPT2DoubleHeadsModel if is_gpt2 else \
            OpenAIGPTDoubleHeadsModel

        self.tokenizer = tokenizer_class.from_pretrained(
            self.hparams.model_checkpoint)
        self.model = model_class.from_pretrained(self.hparams.model_checkpoint)
        # Add special tokens if they are not already added
        add_special_tokens_(self.model, self.tokenizer)

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = self(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        mc_token_ids=mc_token_ids,
                                        mc_labels=mc_labels,
                                        labels=labels)
        loss = lm_loss * self.hparams.lm_coef + mc_loss * self.hparams.mc_coef
        tensorboard_logs = {
            'train_loss': loss,
            'train_lm_loss': lm_loss,
            'train_mc_loss': mc_loss
        }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
        lm_logits, mc_logits, *_ = self.model(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              mc_token_ids=mc_token_ids)
        lm_logits_flat_shifted = \
            lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        labels_flat_shifted = labels[..., 1:].contiguous().view(-1)
        lm_loss = self.criterion(lm_logits_flat_shifted, labels_flat_shifted)
        mc_loss = self.criterion(mc_logits, mc_labels)
        loss = lm_loss * self.hparams.lm_coef + mc_loss * self.hparams.mc_coef
        mc_preds = mc_logits.argmax(-1)
        n_correct_pred = torch.sum(mc_preds == mc_labels)
        return {
            'val_loss': loss,
            'val_lm_loss': lm_loss,
            'val_mc_loss': mc_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(mc_preds)
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_lm_loss = torch.stack([x['val_lm_loss'] for x in outputs]).mean()
        avg_mc_loss = torch.stack([x['val_mc_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / \
                  sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_ppl': avg_loss.exp(),
            'val_lm_loss': avg_lm_loss,
            'val_mc_loss': avg_mc_loss,
            'val_acc': val_acc
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr,
                          correct_bias=True)
        return optimizer

    def prepare_data(self):
        self.personachat = get_dataset(self.tokenizer,
                                       self.hparams.dataset_path,
                                       self.hparams.dataset_cache)

    def setup(self, stage):
        logging.info("Build inputs and labels")
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        for dataset_name, dataset in self.personachat.items():
            num_candidates = len(dataset[0]["utterances"][0]["candidates"])
            if num_candidates > 0 and dataset_name == 'train':
                num_candidates = min(num_candidates, num_candidates)
            for dialog in dataset:
                persona = dialog["personality"].copy()
                for _ in range(self.hparams.personality_permutations):
                    for utterance in dialog["utterances"]:
                        history = utterance["history"] \
                            [-(2 * self.hparams.max_history + 1):]
                        for j, candidate in enumerate(
                                utterance["candidates"][-num_candidates:]):
                            labels = bool(j == num_candidates - 1)
                            instance = build_input_from_segments(persona,
                                                                 history,
                                                                 candidate,
                                                                 self.tokenizer,
                                                                 labels)
                            for input_name, input_array in instance.items():
                                datasets[dataset_name][input_name].append(
                                    input_array)
                        datasets[dataset_name]["mc_labels"].append(
                            num_candidates - 1)
                        datasets[dataset_name]["n_candidates"] = num_candidates
                    # permuted personalities
                    persona = [persona[-1]] + persona[:-1]

        logging.info("Pad inputs and convert to Tensor")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset in datasets.items():
            dataset = pad_dataset(dataset,
                                  padding=self.tokenizer.convert_tokens_to_ids(
                                      SPECIAL_TOKENS[-1]))
            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(dataset[input_name])
                if input_name != "mc_labels":
                    tensor = tensor.view((-1, datasets[dataset_name][
                        "n_candidates"]) + tensor.shape[1:])
                tensor_datasets[dataset_name].append(tensor)

        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        logging.info(f"Train dataset (Batch, Candidates, Seq length): "
                     f"{self.train_dataset.tensors[0].shape}")
        logging.info(f"Valid dataset (Batch, Candidates, Seq length): "
                     f"{self.valid_dataset.tensors[0].shape}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True
                          )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--dataset_path", type=str,
                            default="personachat_self_original.debug.json",
                            help="Path or url of the dataset. If empty "
                                 "download "
                                 "from S3.")
        parser.add_argument("--dataset_cache", type=str,
                            default='./dataset_cache',
                            help="Path or url of the dataset cache")
        parser.add_argument("--model_checkpoint", type=str,
                            default="gpt2",
                            help="Path, url or short name of the model")
        parser.add_argument("--num_candidates", type=int, default=2,
                            help="Number of candidates for training")
        parser.add_argument("--max_history", type=int, default=2,
                            help="Number of previous exchanges to keep in "
                                 "history")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Batch size")
        parser.add_argument("--lr", type=float, default=6.25e-5,
                            help="Learning rate")
        parser.add_argument("--lm_coef", type=float, default=1.0,
                            help="LM loss coefficient")
        parser.add_argument("--mc_coef", type=float, default=1.0,
                            help="Multiple-choice loss coefficient")
        parser.add_argument("--personality_permutations", type=int, default=1,
                            help="Number of permutations of personality "
                                 "sentences")

        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of workers in data loader')
        return parser


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class
    and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [
            x +
            [padding if name != "labels" else IGNORE_INDEX] * (max_l -
                                                                  len(x))
            for x in dataset[name]
        ]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not
    already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens +
                                                     num_added_tokens)


def build_input_from_segments(persona, history, reply, tokenizer,
                              labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last
    reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + \
               history + \
               [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + \
               [
                   [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
                   for i, s in enumerate(sequence[1:])
               ]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1
                                  for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["labels"] = [IGNORE_INDEX] * len(instance["input_ids"])
    if labels:
        instance["labels"] = ([IGNORE_INDEX] * sum(len(s)
                                                      for s in sequence[:-1]
                                                      )
                                 ) + \
                                [IGNORE_INDEX] + sequence[-1][1:]
    return instance
