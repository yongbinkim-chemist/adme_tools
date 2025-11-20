# adme_tools

A command-line wrapper for training, fine-tuning, and running AI/ML models for **ADME property prediction**  
(e.g., Caco-2 permeability, BBB penetration, CYP3A4 inhibition, half-life, and more later).

Written by **Yongbin Kim, Nov 2025**.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yongbinkim-chemist/adme_tools.git
cd adme_tools
```

### 2. Create the Conda environment
```bash
conda env create -f environment.yml
conda activate my-rhino-env
```
### 3. Install the package (editable mode)
```bash
pip install -e .
```

`run_adme` is now available as a global command:
```bash
run_adme --help
```

## Command-line Arguments
Below are all CLI options supported by `run_adme`, exactly as defined in `parse_args()`.

| Argument | Type | Default | Choices | Description |
|----------|------|---------|----------|-------------|
| `-p`, `--property` | `str` | `Caco2_Wang` | `Caco2_Wang`, `bbb_martins`, `CYP3A4_Veith`, `Half_Life_Obach` | ADME endpoint to train/predict. |
| `-t`, `--task` | `str` | `train` | `train`, `predict`, `tune` | Task: train / fine-tune / predict. |
| `-d`, `--descriptor` | `str` | `maccs` | `maccs`, `mpnn` | Molecular descriptor / model type. |
| `--smiles_col` | `str` | `Drug` | – | Column name containing SMILES strings. |
| `--target_col` | `str` | `Y` | – | Column containing ADME numeric labels. |
| `--split_method` | `str` | `scaffold` | `scaffold`, `random` | How to split training/validation set. |
| `-lr`, `--learning_rate` | `float` | `0.01` | – | Learning rate for optimization. |
| `--batch` | `int` | `16` | – | Mini-batch size. |
| `--epochs` | `int` | `5` | – | Number of epochs. |
| `-pd`, `--param_dir` | `str` | `None` | – | Path to existing model checkpoint (`.pth`) used for tuning or prediction. |
| `-data`, `--data_dir` | `str` | `None` | – | Path to input CSV file (for training, tuning, or prediction). |
| `--seed` | `int` | `None` | – | Optional random seed. |
| `--input-shape` | `int` | `167` | – | Input feature dimension (MACCS = 167). |
| `--hidden-units` | `list[int]` | `[128, 64, 32]` | – | Hidden layer sizes for MACCS feed-forward network. |
| `--output-shape` | `int` | `1` | – | Output dimension (regression = 1). |

### Notes
- When `--task tune` or `--task predict`, both `--param_dir` and `--data_dir` must be provided.
- For `--descriptor mpnn`, Chemprop-based MPNN model is used.
- For `--descriptor maccs`, MACCS+FCN neural network is used.

## Data Format

This tool expects input datasets in **CSV format**.  
Depending on the task (`train`, `tune`, `predict`), the required columns differ.

---

### Training / Fine-tuning Data

For `--task train` and `--task tune`, your CSV **must include both**:

| Column | Description |
|--------|-------------|
| `--smiles_col` (default: `Drug`) | SMILES string for the molecule. |
| `--target_col` (default: `Y`) | Experimental ADME value (float). |

#### Example (`dataset/caco2_train.csv`)

```csv
Drug,Y
CC(=O)OC1=CC=CC=C1C(=O)O,-4.57
CCN(CC)CCCC(C)NC1=NC2=CC=CC=C2N1,-5.49
CCOC(=O)c1ccc(O)cc1,-4.92
```

## Prediction data
Only SMILES column is required.

## Example Usage

This section shows how to use the `run_adme` command for:
- **Training** a new ADME model  
- **Fine-tuning** a pretrained model  
- **Predicting** ADME properties for new molecules  

All explanations, bullets, and descriptions below are fully included.

---

## 1. Train a New ADME Model from Scratch

This command trains a MACCS-based neural network to predict the `Caco2_Wang` property:

```bash
run_adme \
  -p Caco2_Wang \
  -t train \
  -d maccs \
  -lr 0.01 \
  --smiles_col Drug \
  --target_col Y \
  --batch 16 \
  --epochs 5
```

### This Mode:
- Reads training data from `--data_dir` if provided, or uses the default dataset specified in your code
- Uses MACCS fingerprints (`-d maccs`) unless `mpnn` is selected
- Automatically performs a train/validation/test split:
    - *scaffold (default)* or
    - *random* if `--split_method random`
- Trains a regression model for the selected property (`-p`)
- Saves the trained model checkpoint

## 2. Fine-tune an Existing Pretrained Model (Transfer Learning)
Use this to adapt a pretrained ADME model (e.g., from public data) to a small project-specific dataset.

```bash
run_adme \
  -p Caco2_Wang \
  -t tune \
  -d maccs \
  -lr 0.01 \
  --smiles_col Drug \
  --batch 16 \
  --epochs 5 \
  -pd models/maccs.pth \
  -data dataset/caco2_valid.csv
```
### This mode:
- Re-runs training using the new dataset while initializing parameters from the pretrained model.
- Useful for:
    - project-specific chemical series
    - new in-house ADME assays
    - small datasets requiring transfer learning
Requirements for this mode:

| Argument | Description |
|----------|-------------|
| `-pd / --param_dir` | Path to existing model checkpoint (`.pth`) used for tuning or prediction. |
| `-data / --data_dir` | Path to input CSV file (for training, tuning, or prediction). |

## 3. Run Prediction on New Molecules
This command loads a trained/fine-tuned model and predicts ADME properties for new SMILES.

```bash
run_adme \
  -p Caco2_Wang \
  -t predict \
  -d maccs \
  --smiles_col Drug \
  -pd models/maccs-tuned.pth \
  -data dataset/caco2_test.csv
```

### This mode:
- Loads the model checkpoint from `--param_dir`
- Reads SMILES from `--data_dir`
- Converts each molecule to MACCS or MPNN features
- Predicts the ADME property specified by `--property`
- Outputs predictions into a CSV file (if implemented in your pipeline)
- Target values (`Y`) in the dataset are ignored for prediction mode

## ⚠️Prototype Notice

This project is still in an **early prototype stage**.  
Many components—including data loaders, training pipelines, logging, model saving/loading,  
and MPNN integration—are under active development and will continue to evolve.

Please be aware of the following:

- The codebase is **not yet fully optimized** for robustness, scalability, or production use.
- Additional ADME endpoints, descriptors, and model architectures will be added in future updates.
- Error handling, documentation, and CLI behaviors may change as the project matures.
- Certain features (e.g., checkpoint saving, result reporting, hyperparameter tuning, and Chemprop integration) may be **incomplete** or require further refinement.
- Contributions, issues, and feedback are welcome as the toolkit continues to grow.

This repository is intended as a **working prototype** to support ongoing ADME/ML research and experimentation,  
and should be treated as an evolving codebase rather than a finalized library.

## Author
Yongbin Kim
Email: `chem.yongbin@gmail.com`
Description: Tools for computational ADME prediction (MACCS + FCN, Chemprop MPNN, PyTorch Lightning).
