import logging
import json
from argparse import ArgumentParser, Namespace
from itertools import cycle
from pathlib import Path
from random import choice, randrange
from typing import List
from collections import defaultdict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import AdamW, GPT2DoubleHeadsModel, GPT2Tokenizer
from tqdm.auto import tqdm
from torchtext.data.metrics import bleu_score


IGNORE_INDEX = -100
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
SYSTEM = "<system>"
USER = "<user>"
SPECIAL_TOKENS = [BOS, EOS, SYSTEM, USER, PAD]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "additional_special_tokens": [SYSTEM, USER],
}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]

TASK_ORIENTED_DATASETS = ["sgd", "multiwoz"]


class ConditionalLM(LightningModule):

    _SPLITS = ["train", "valid", "test"]

    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_checkpoint)
        self.model = GPT2DoubleHeadsModel.from_pretrained(self.hparams.model_checkpoint)
        # Add special tokens if they are not already added
        add_special_tokens_(self.model, self.tokenizer)

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            mc_labels=mc_labels,
            labels=labels,
        )
        loss = lm_loss * self.hparams.lm_coef + mc_loss * self.hparams.mc_coef
        tensorboard_logs = {
            "train_loss": loss,
            "train_lm_loss": lm_loss,
            "train_mc_loss": mc_loss,
        }
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
        lm_loss, mc_loss, lm_logits, mc_logits, *_ = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            labels=labels,
            mc_labels=mc_labels,
        )
        loss = lm_loss * self.hparams.lm_coef + mc_loss * self.hparams.mc_coef
        mc_preds = mc_logits.argmax(-1)
        n_correct_pred = torch.sum(mc_preds == mc_labels)

        active_lm_mask = labels[:, -1, ...] != IGNORE_INDEX
        masked_lm_logits = [
            lm_logit[mask]
            for mask, lm_logit in zip(active_lm_mask, lm_logits[:, -1, ...])
        ]
        masked_lm_labels = [
            lm_label[mask] for mask, lm_label in zip(active_lm_mask, labels[:, -1, ...])
        ]
        masked_lm_preds = [l.argmax(-1) for l in masked_lm_logits]
        lm_pred_texts = [
            self.tokenizer.decode(pred, True).split() for pred in masked_lm_preds
        ]
        lm_label_texts = [
            [self.tokenizer.decode(label, True).split()] for label in masked_lm_labels
        ]
        bleu = torch.tensor(
            [bleu_score(lm_pred_texts, lm_label_texts)], device=self.device
        )

        return {
            "val_loss": loss,
            "val_lm_loss": lm_loss,
            "val_mc_loss": mc_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(mc_preds),
            "bleu": bleu,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_lm_loss = torch.stack([x["val_lm_loss"] for x in outputs]).mean()
        avg_mc_loss = torch.stack([x["val_mc_loss"] for x in outputs]).mean()
        val_acc = torch.true_divide(
            torch.stack([x["n_correct_pred"] for x in outputs]).sum(),
            sum(x["n_pred"] for x in outputs),
        )
        avg_bleu = torch.stack([x["bleu"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_ppl": avg_loss.exp(),
            "val_lm_loss": avg_lm_loss,
            "val_mc_loss": avg_mc_loss,
            "val_acc": val_acc,
            "val_bleu": avg_bleu,
        }
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        output = self.validation_epoch_end(*args, **kwargs)
        output = {k.replace("val", "test"): v for k, v in output.items()}
        output["log"] = {k.replace("val", "test"): v for k, v in output["log"].items()}
        return output

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr, correct_bias=True
        )
        return optimizer

    def prepare_split_data(self, split: str):
        assert split in {"train", "valid", "test"}
        datasets = []
        pad_id = self.tokenizer.convert_tokens_to_ids(PAD)
        if not hasattr(self.hparams, "dataset_paths"):
            self.hparams.dataset_paths = [self.hparams.dataset_path]
        for dataset_path in self.hparams.dataset_paths:
            tensor_dataset_cache_path = Path(
                f"tensor_dataset_cache_{split}_"
                f"{self.hparams.model_checkpoint}_"
                f"{dataset_path}.pt".replace("/", "_SLASH_")
            )
            if tensor_dataset_cache_path.exists():
                logging.info(
                    f"{split} tensor dataset loaded from" f"{tensor_dataset_cache_path}"
                )
                datasets.append(torch.load(tensor_dataset_cache_path.open("rb")))
            else:
                dataset = json.loads(Path(dataset_path).read_text())[split]

                is_task_oriented = any(
                    name.lower() in dataset_path.lower()
                    for name in TASK_ORIENTED_DATASETS
                )

                logging.info("Build inputs and labels")
                processed_dataset = defaultdict(list)
                num_candidates = 2
                for dial_idx, dialog in tqdm(dataset.items(), desc=split):

                    def _sample_random_response():
                        _first = True
                        while (
                            _first
                            or random_dial_idx == dial_idx
                            or len(random_dial) < 2
                        ):
                            random_dial_idx = choice(list(dataset.keys()))
                            random_dial = dataset[random_dial_idx]
                            _first = False
                        random_turn_idx = randrange(0, len(random_dial) // 2)
                        return random_dial[random_turn_idx * 2 + 1]

                    for i, response in enumerate(dialog):
                        if i % 2 == 0:
                            # skip user turn
                            continue
                        history = dialog[i - (2 * self.hparams.max_history + 1) : i]
                        if is_task_oriented:
                            if i + 1 >= len(dialog):
                                # SGD dataset have even number of turns
                                # i.e. the last speaker in dialog may not be user
                                continue
                            else:
                                next_user_utterance = dialog[i + 1]
                        else:
                            next_user_utterance = ""
                        random_utterance = _sample_random_response()
                        for labels, candidate in zip(
                            [False, True], [random_utterance, response]
                        ):
                            instance = build_input_from_segments(
                                history,
                                next_user_utterance,
                                candidate,
                                labels,
                                self.tokenizer,
                            )
                            for input_name, input_array in instance.items():
                                processed_dataset[input_name].append(input_array)
                        processed_dataset["mc_labels"].append(num_candidates - 1)
                        processed_dataset["n_candidates"] = num_candidates

                logging.info("Pad inputs and convert to Tensor")
                tensor_dataset = []
                processed_dataset = pad_dataset(processed_dataset, pad_id)
                for input_name in MODEL_INPUTS:
                    tensor = torch.tensor(processed_dataset[input_name])
                    if input_name != "mc_labels":
                        tensor = tensor.view(
                            (-1, processed_dataset["n_candidates"]) + tensor.shape[1:]
                        )
                    tensor_dataset.append(tensor)

                dataset = TensorDataset(*tensor_dataset)
                datasets.append(dataset)

                torch.save(dataset, str(tensor_dataset_cache_path))
                logging.info(
                    f"{split.capitalize()} tensor dataset saved to "
                    f"{tensor_dataset_cache_path}"
                )
            logging.info(
                f"{dataset_path} {split} dataset "
                f"(Batch, Candidates, Seq length): "
                f"{datasets[-1].tensors[0].shape}"
            )
        concat_dataset = self.pad_concat_datasets(datasets, pad_id)
        setattr(self, f"{split}_dataset", concat_dataset)

    def prepare_data(self):
        for split in self._SPLITS:
            self.prepare_split_data(split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def pad_concat_datasets(
        datasets: List[TensorDataset], pad_id: int
    ) -> ConcatDataset:
        index = {0, 2, 4}
        max_len = max(dataset.tensors[0].size(-1) for dataset in datasets)
        for dataset in datasets:
            tensors = []
            cur_len = dataset.tensors[0].size(-1)
            num_pad = max_len - cur_len
            for i, tensor in enumerate(dataset.tensors):
                if i in index:
                    size = tensor.shape[:-1] + (num_pad,)
                    padded = torch.full(
                        size, pad_id, device=tensor.device, dtype=tensor.dtype
                    )
                    tensors.append(torch.cat((tensor, padded), -1))
                else:
                    tensors.append(tensor)
            dataset.tensors = tuple(tensors)
        concat_dataset = ConcatDataset(datasets)
        return concat_dataset

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            "--dataset_paths",
            type=str,
            nargs="+",
            default=["data/multiwoz2.1.processed.json"],
            help="Path of the dataset.",
        )
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="microsoft/DialoGPT-small",
            help="Path, url or short name of the model",
        )
        parser.add_argument(
            "--max_history",
            type=int,
            default=2,
            help="Number of previous exchanges to keep in " "history",
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
        parser.add_argument(
            "--lm_coef", type=float, default=2.0, help="LM loss coefficient"
        )
        parser.add_argument(
            "--mc_coef",
            type=float,
            default=1.0,
            help="Multiple-choice loss coefficient",
        )

        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers in data loader",
        )
        return parser


def pad_dataset(dataset, padding=0):
    """Pad the dataset. This could be optimized by defining a Dataset class
    and padding at the batch level, but this is simpler."""
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [
            x + [padding if name != "labels" else IGNORE_INDEX] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset


def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not
    already been added."""
    orig_num_tokens = len(tokenizer.encoder)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_input_from_segments(
    history: List[str],
    next_user_utterance: str,
    reply: str,
    is_ground_truth_reply: bool,
    tokenizer: GPT2Tokenizer,
):
    """Build a sequence of input from 3 segments: history, next usr reply and
    next system reply."""

    def tokenize_to_ids(x: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))

    bos, eos, user, system = tokenizer.convert_tokens_to_ids([BOS, EOS, USER, SYSTEM])
    history: List[List[int]] = [
        tokenize_to_ids(type + " " + turn)
        for turn, type in zip(history, cycle([USER, SYSTEM]))
    ]
    next_user_utterance_ids: List[int] = tokenize_to_ids(
        USER + " " + next_user_utterance
    )
    reply: List[int] = tokenize_to_ids(SYSTEM + " " + reply)
    sequence = [bos]
    token_type_ids = [user]
    for seq, type in zip(history, cycle([user, system])):
        assert type in [user, system]
        sequence.extend(seq)
        token_type_ids.extend([type] * len(seq))
    if next_user_utterance.strip():
        sequence.extend(next_user_utterance_ids)
        token_type_ids.extend([user] * len(next_user_utterance_ids))
    num_prompt = len(sequence) + 1
    sequence.extend(reply)
    token_type_ids.extend([system] * len(reply))
    sequence.append(eos)
    token_type_ids.append(system)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_type_ids,
        "mc_token_ids": len(sequence) - 1,
        "labels": [IGNORE_INDEX] * num_prompt + sequence[num_prompt:]
        if is_ground_truth_reply
        else [IGNORE_INDEX] * len(sequence),
    }
    assert (
        len(instance["input_ids"])
        == len(instance["token_type_ids"])
        == len(instance["labels"])
    )
    return instance
