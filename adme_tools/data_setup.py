import pandas as pd

from typing import List, Tuple

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

import torch

import chemprop

from tdc.single_pred import ADME

def load_data(prop: str,
              smi_col: str,
              method: str="scaffold",
              frac: List[float]=[0.7, 0.1, 0.2]) -> Tuple[pd.DataFrame]:

    data = ADME(name=prop)

    # train / valid / test
    splits = data.get_split(method=method, frac=frac) # random
    train_df, valid_df, test_df = splits.values()

    # train_df.to_csv(f"{prop}_train.csv", index=False)
    # valid_df.to_csv(f"{prop}_valid.csv", index=False)
    # test_df.to_csv(f"{prop}_test.csv", index=False)

    train_df['molecule'] = train_df[smi_col].apply(Chem.MolFromSmiles)
    valid_df['molecule'] = valid_df[smi_col].apply(Chem.MolFromSmiles)
    test_df['molecule'] = test_df[smi_col].apply(Chem.MolFromSmiles) 

    return train_df, valid_df, test_df

def maccs_create_dataloader(df: pd.DataFrame,
                            shuffle: bool,
                            target_col: str,
                            batch_size: int,
                            pred: bool=False) -> torch.utils.data.DataLoader:
    
    df['MACCS'] = df['molecule'].apply(MACCSkeys.GenMACCSKeys)
    X = torch.tensor(df['MACCS'],
                     dtype=torch.float32)

    y = torch.zeros((len(X), 1), dtype=torch.float32)
    if not pred:
        y = torch.tensor(df[target_col],
                         dtype=torch.float32).unsqueeze(1) 

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader

def mpnn_create_dataloader(df: pd.DataFrame,
                           shuffle: bool,
                           mol_col: str,
                           target_col: str,
                           batch_size: int,
                           normalize: bool,
                           pred: bool=False):

    smiles = df[mol_col].tolist()
    targets = np.zeros(len(smiles), dtype="float32")
    if not pred:
        targets = df[target_col].values.astype("float32")

    datapoints = [chemprop.data.MoleculeDatapoint.from_smi(s, y=[t]) for s, t in zip(smiles, targets)]

    featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer() # <- graph featurization
    dataset = chemprop.data.MoleculeDataset(datapoints, featurizer=featurizer)

    if normalize:
        from joblib import dump, load
        output_scaler = dataset.normalize_targets()
        dump(output_scaler, "checkpoints/output_scaler.pkl")

    dataloader = chemprop.data.build_dataloader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=1)
    return dataloader
