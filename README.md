# Dual-Path Cognitive Irony Detector (DCID)

**DCID (Dual-Path Cognitive Irony Detector)** is a Chinese irony/sarcasm detection framework implemented in PyTorch.  
The current codebase provides an end-to-end training & evaluation pipeline with repeated experiments over multiple random seeds.

> Repository structure is intentionally kept simple: `main.py` as the entry script + a local `model/` folder (optional) for offline caching.

---

## Repository Structure

- `main.py`  
  Main entry for loading data, training, validation threshold search, and test evaluation.
- `model/`  
  Optional local HuggingFace cache / offline model directory.  
- `model_weights.pt`  
  Trained DCID model weights.

---

## Environment Setup

### 1) Create a Python environment (recommended)

    conda create -n dcid python=3.10 -y
    conda activate dcid

### 2) Install dependencies

    pip install torch transformers scikit-learn pandas numpy

Notes:
- Install the correct `torch` build for your CUDA version if you use GPU.
- The script automatically selects CUDA when available.

---

## Dataset Format

DCID expects a CSV file containing the following columns:

- `评论内容` : the comment text (string)
- `话题` : the topic (string)
- `讽刺性[0/1]` : label (0 or 1)

---

## Model Initialization (BERT)

The code uses a Chinese BERTWWM model via HuggingFace `transformers`.

### Recommended

Set the model path in `main.py` as a model name:

    BERT_PATH = "hfl/chinese-bert-wwm-ext"

This will download the model automatically on first run.

### Offline / local directory

If you have downloaded model files locally, you can set:

    BERT_PATH = "./model/<your_local_model_dir>"

---

## Run Training & Evaluation

Default run (from the repository root):

    python main.py

What the script does:
- Loads CSV data
- Splits into train/val/test (70/10/20 approximately via two-stage split)
- Trains DCID with early stopping
- Searches the best threshold on the validation set (for F1)
- Evaluates on test set
- Repeats experiments across a list of random seeds
- Saves results to `experiment_results.csv`

---

## Outputs

- `experiment_results.csv`  
  Contains metrics for each random seed experiment:
  - validation: accuracy / precision / recall / F1 / AUC + best threshold
  - test: accuracy / precision / recall / F1 / AUC
  - training time

---

## Common Issues

### Download problems for models

If your environment cannot access HuggingFace, use an offline directory:
- download the model elsewhere
- copy it under `model/`
- set `BERT_PATH` to that local directory

---

## License

This project is released under the **Apache-2.0 License** (see `LICENSE`).

---

## Citation

If you use DCID in your research, please cite the repository:

    @software{dcid_2026,
      title  = {Dual-Path Cognitive Irony Detector (DCID)},
      author = {Dylan Wang},
      year   = {2026},
      url    = {https://github.com/DylanWang2094/DCID}
    }

---

## Contact

For questions or collaboration, please open an issue on GitHub.
