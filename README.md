
##  [Experiemntal Logging](https://app.wandb.ai/ytlin/controllable-response?workspace=user-ytlin)

## Installation

### Develop
```bash
pip install -r requirements.txt
```

### Deploy
```bash
docker image build -t transition:latest .
```

## Data
```bash
bash prepare_data.sh
```

The processed file is stores at `data/<dataset_name>.processed.json`

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


## Training

The training script can be used in single GPU or multi GPU settings:

```bash
python trainer.py -h  # To see options for training
```

## Interactive Demo

CHECKPOINT could be local file dir to `save_pretrained` in `transformers`  output 
or `<user/model_name>` on HuggingFace Model Hub

eg. `ytlin/verr5re0` or `ytlin/1pm2c7qw_6`

```bash
python demo.py <checkpoint>
```

## Generate Sample
```bash
python generate_response.py <checkpoint> <input dialogues path> <output path>
# eg
python generate_response.py ytlin/1klqb7u9_35 ./data/human_eval/chit_to_task_cue.txt ./data/human_eval/chit_to_task_cue.txt.gen
```

### Docker

```bash
docker run --rm -it -e CHECKPOINT=<checkpoint> transition:latest
```