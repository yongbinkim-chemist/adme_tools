# adme_tools

A command-line wrapper for training, fine-tuning, and running AI/ML models for **ADME property prediction**  
(e.g., Caco-2 permeability, BBB penetration, CYP3A4 inhibition, half-life, and more later).

Written by **Yongbin Kim, Nov 2025**.

---

## 📦 Installation

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
