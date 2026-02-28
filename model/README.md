# Model Directory (Local)

This folder is used for local HuggingFace model cache (optional).

## Recommended (Online)
Set `BERT_PATH` in `main.py` to:
- `hfl/chinese-bert-wwm-ext`

The model will be downloaded automatically by `transformers`.

## Offline (Local path)
If you already have the model files locally, place them under `model/` and set:
- `BERT_PATH = "./model/<your_local_model_dir>"`