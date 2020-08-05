## Installation

### Develop
To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
pip install -r requirements.txt
```

### Deploy
```bash
docker image build -t transition:latest .
```

## Preprocess Data and Format

Using `persona-chat` as an example.

Other datasets can be run mostly the same way.

```bash
bash download_personachat.sh
python preprocess_persona-chat.py
```

The processed file is stores at `data/persona.processed.json`

Json format:
```json5
{
  "train": {
    "train_0": [ "Other's utterance", "Our turn", ... ],
    ...
  },
  "valid": {...},
  "test": {...}
}
```


## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python trainer.py -h  # To see options for training
```

## Interactive Evaluation

```bash
python test.py <checkpoint_path> <hparams_file> --interactive
```

### Docker

```bash
docker run --rm -it transition:latest
# To specify checkpoints
docker run --rm -it -e CHECKPOINT=<checkpoint_path> -e HPARAMS=<hparams_file> transition:latest
```