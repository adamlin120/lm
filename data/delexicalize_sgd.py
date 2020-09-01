"""
This file is modified from https://github.com/mtkresearch/salsa/blob/ductile_toward_task/ductile_dialogue/scripts/delex.py
"""
import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from inflect import engine
from word2number import w2n
from tqdm.auto import tqdm


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Delexicalise data set")
    parser.add_argument('dataset_folder', help="folder where the json files are", type=Path)
    return parser.parse_args()


def delexicalize(utterance, slot_value, slot_name):
    return re.sub(r'\b' + slot_value + r'\b', slot_name + "_", utterance)


def scan_replace(
        utterance: str,
        slot_content,
        x,
        inflect_engine: engine
) -> str:
    for y in slot_content:  # A slot content is a list, so we need to iterate on its elements
        y = y.replace("+", "00").replace("$", "dollars")
        # basic substitution, but exclude True and False string (coming from binary slots)
        if y != "True" and y != "False":
            utterance = delexicalize(utterance, y, x)

        # Deal with case in which the slot value is a digit and the same number is in words in the utterance
        if y.isnumeric():
            y_word = inflect_engine.number_to_words(y)
            utterance = delexicalize(utterance, y_word, x)

        # Deal with case in which the slot value is a number in words and the same number is in digits in the utterances
        try:
            y_digit = str(w2n.word_to_num(y))
            utterance = delexicalize(utterance, y_digit, x)
        except ValueError:
            pass
    return utterance


def main(args: Namespace):
    inflect_engine = engine()

    for split in ['train', 'dev', 'test']:
        folder = args.dataset_folder / split
        dump = []

        for path in tqdm(folder.glob('dialogues_*.json')):
            dialogues = json.loads(path.read_text())
            for j, dialog in enumerate(dialogues):
                dump.append("[STARTCONVERSATION]")
                for k, turn in enumerate(dialog["turns"]):
                    utterance = turn['utterance'].replace("+", "00").replace("$", "dollars")
                    frames = turn['frames'][0]
                    if k % 2 == 0:
                        slot_values = frames['state']['slot_values']
                        for x in slot_values:
                            utterance = scan_replace(utterance, slot_values[x], x, inflect_engine)
                    else:
                        for action in frames["actions"]:
                            utterance = scan_replace(utterance, action["values"], action["slot"], inflect_engine)

                    dump.append(utterance)
        output = args.dataset_folder / f'{split}.txt'
        output.write_text('\n'.join(dump))


if __name__ == '__main__':
    main(_parse_args())
