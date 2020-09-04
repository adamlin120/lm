import logging
from argparse import ArgumentParser, Namespace
from typing import List, Dict
from itertools import cycle
from pathlib import Path

import torch
import ipdb
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from module import build_input_from_segments, SYSTEM, PAD, EOS

logging.basicConfig(level=logging.INFO)

ENCODING = "cp1252"

SOD = "[STARTCONVERSATION]"
INTERLOCUTOR_A = "A:"
INTERLOCUTOR_B = "B:"
CUE = "C:"


def _is_valid_turn_start(turn: str) -> bool:
    return (
        turn.startswith(INTERLOCUTOR_A)
        or turn.startswith(INTERLOCUTOR_B)
        or turn.startswith(CUE)
    )


def _is_alternating_interlocutors(dialogue: List[str]) -> bool:
    roles = [turn.split()[0] for turn in dialogue]
    return all(
        role == role_expected
        for role, role_expected in zip(roles, cycle([INTERLOCUTOR_A, INTERLOCUTOR_B]))
    )


def remove_speaker_tag(turn: str) -> str:
    return turn.strip(INTERLOCUTOR_A).strip(INTERLOCUTOR_B).strip(CUE)


def prepare_instance(
    tokenizer: GPT2Tokenizer, device: torch.device, history: List[str], cue: str
) -> Dict[str, torch.Tensor]:
    instance = build_input_from_segments(
        history=history,
        next_user_utterance=cue,
        reply="",
        is_ground_truth_reply=False,
        tokenizer=tokenizer,
    )
    instance.pop("mc_token_ids")
    instance.pop("labels")
    instance["input_ids"].pop(-1)
    instance["token_type_ids"].pop(-1)

    tensor_instance = {k: torch.tensor([v], device=device) for k, v in instance.items()}
    tensor_instance["input_ids"] = tensor_instance["input_ids"].repeat(2, 1)
    tensor_instance["token_type_ids"] = tensor_instance["token_type_ids"].repeat(2, 1)
    return tensor_instance


def load_dialogues(
    path: Path,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    dialogues: List[List[str]] = [
        dialogue.strip().splitlines()
        for dialogue in path.read_text(encoding=ENCODING).strip().split(SOD)
        if dialogue.strip()
    ]
    assert all(
        _is_valid_turn_start(turn) for dialogue in dialogues for turn in dialogue
    ), ValueError("Invalid turn start!")

    has_cue = any(turn.startswith(CUE) for dialogue in dialogues for turn in dialogue)

    assert (not has_cue) or all(dialogue[-1].startswith(CUE) for dialogue in dialogues)
    assert all(
        _is_alternating_interlocutors(dialogue[:-1] if has_cue else dialogue)
        for dialogue in dialogues
    )

    parsed_dialogues = [
        {
            "cue": remove_speaker_tag(dialogue[-1] if has_cue else ""),
            "history": list(
                map(remove_speaker_tag, dialogue[:-1] if has_cue else dialogue)
            ),
        }
        for dialogue in dialogues
    ]

    return [
        prepare_instance(tokenizer, device, **dialogue) for dialogue in parsed_dialogues
    ]


def write_dialogues_with_response(
    path: Path,
    output_path: Path,
    responses: List[str],
) -> None:
    dialogues: List[str] = [
        dialogue.strip()
        for dialogue in path.read_text(encoding=ENCODING).strip().split(SOD)
        if dialogue.strip()
    ]
    assert len(dialogues) == len(responses)
    dialogues_with_response: List[str] = [
        SOD + "\n" + dialogue + "\n" + "S: " + response.strip()
        for dialogue, response in zip(dialogues, responses)
    ]
    output_path.write_text("\n".join(dialogues_with_response))


def main(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval().to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint_path)

    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(PAD)

    dialogues = load_dialogues(args.dialogues, tokenizer, device)
    responses: List[str] = []
    for tensor_instance in dialogues:
        preds = model.generate(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=512,
            # top_k=50,
            # top_p=0.95,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1,
            **tensor_instance
        )
        preds_txt = [tokenizer.decode(pred) for pred in preds]
        pred_txt = preds_txt[0]
        sys_reply_start = pred_txt.rfind(SYSTEM)
        sys_reply = pred_txt[sys_reply_start:].strip(SYSTEM)
        responses.append(sys_reply)

    write_dialogues_with_response(args.dialogues, args.output_path, responses)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("dialogues", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main(parse_args())
