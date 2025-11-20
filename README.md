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
