import torch

import pandas as pd

import rdkit
from rdkit import Chem

import sklearn
from sklearn.model_selection import train_test_split

from .data_setup import load_data
from .pipelines import get_pipeline

# from engine import run_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_train(args):
    
    DESCRIPTOR = args.descriptor
    PROP = args.property
    LEARNING_RATE = args.learning_rate    
    EPOCHS = args.epochs
    SMILES_COL = args.smiles_col
    SPLIT_METHOD = args.split_method

    train_df,valid_df,test_df = load_data(prop=PROP,
                                          smi_col=SMILES_COL,
                                          method=SPLIT_METHOD,
                                          frac=[0.7,0.1,0.2])
    
    pipeline = get_pipeline(args)
    
    model = pipeline.model_cls
    create_dataloader = pipeline.create_dataloader
    loader_params = pipeline.loader_params
    model_params = pipeline.model_params
    torch_run = pipeline.torch_run
    trainer = pipeline.trainer
    tester = pipeline.tester

    train_dataloader = create_dataloader(df=train_df,
                                         shuffle=True,
                                         **loader_params)

    valid_dataloader = create_dataloader(df=valid_df,
                                         shuffle=False,
                                         **loader_params)

    test_dataloader = create_dataloader(df=test_df,
                                        shuffle=False,
                                        **loader_params)
    
    model = model(**model_params)
    loss_fn = None
    optimizer = None
    if torch_run:
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=LEARNING_RATE)
    
    train_results = trainer(model=model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            epochs=EPOCHS,
                            device=device,
                            save=True,
                            optimizer=optimizer,
                            loss_fn=loss_fn)
    
    if DESCRIPTOR == 'maccs':
        test_results, preds, ground_truth = tester(model=model,
                                                   dataloader=test_dataloader,
                                                   device=device,
                                                   loss_fn=loss_fn)
        train_df = pd.DataFrame(train_results)
        test_df = pd.DataFrame(test_results)
        pred_df = pd.DataFrame({"ground_truth": ground_truth.cpu().numpy(),
                                "predicted_value": preds.cpu().numpy()})
        df = pd.concat([train_df, test_df, pred_df], axis=1)
        name = f"{PROP}-{DESCRIPTOR}-results.csv"
        df.to_csv(name, index=False)
    # if DESCRIPTOR == "mpnn":
    #     tester_params = {'dataloaders': test_dataloader}
    #     results = trainer.test(dataloaders=test_dataloader) 

def run_predict(args):
    
    DESCRIPTOR = args.descriptor
    PROP = args.property
    SMILES_COL = args.smiles_col
    DATA_DIR = args.data_dir
    STATE_DICT_DIR = args.param_dir

    test_df = pd.read_csv(DATA_DIR)
    test_df['molecule'] = test_df[SMILES_COL].apply(Chem.MolFromSmiles) 

    pipeline = get_pipeline(args)
    model = pipeline.model_cls
    model_params = pipeline.model_params
    loader_params = pipeline.loader_params
    create_dataloader = pipeline.create_dataloader
    torch_run = pipeline.torch_run
    tester = pipeline.tester
    
    test_dataloader = create_dataloader(df=test_df,
                                        shuffle=False,
                                        **loader_params)
    model = model(**model_params)
    loss_fn = None
    if torch_run:
        loss_fn = torch.nn.MSELoss()
    
    model.load_state_dict(torch.load(STATE_DICT_DIR))
    model.to(device)

    if DESCRIPTOR == 'maccs':
        _, preds, _ = tester(model=model,
                             dataloader=test_dataloader,
                             device=device,
                             loss_fn=loss_fn)

        pred_df = pd.DataFrame({"predicted_value": preds.cpu().numpy()})
        name = f"{PROP}-{DESCRIPTOR}-prediction-results.csv"
        pred_df.to_csv(name, index=False)

def fine_tunning(args):
    
    DESCRIPTOR = args.descriptor
    PROP = args.property
    LEARNING_RATE = args.learning_rate    
    EPOCHS = args.epochs
    SMILES_COL = args.smiles_col
    TARGET_COL = args.target_col
    SPLIT_METHOD = args.split_method
    DATA_DIR = args.data_dir
    STATE_DICT_DIR = args.param_dir

    df = pd.read_csv(DATA_DIR)
    df['molecule'] = df[SMILES_COL].apply(Chem.MolFromSmiles)

    pipeline = get_pipeline(args)
    model = pipeline.model_cls
    create_dataloader = pipeline.create_dataloader
    loader_params = pipeline.loader_params
    model_params = pipeline.model_params
    torch_run = pipeline.torch_run
    trainer = pipeline.trainer

    train_dataloader = create_dataloader(df=df,
                                         shuffle=True,
                                         **loader_params)

    model = model(**model_params)
    loss_fn = None
    optimizer = None
    model.load_state_dict(torch.load(STATE_DICT_DIR))
    model.to(device)

    if torch_run:
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=LEARNING_RATE)
    
    for idx, p in enumerate(model.parameters()):
        if idx < len(model.layers)-1:
            p.requires_grad = False
        # print(p.requires_grad)

    train_results = trainer(model=model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=None,
                            epochs=EPOCHS,
                            device=device,
                            save=True,
                            tune=True,
                            optimizer=optimizer,
                            loss_fn=loss_fn)
    
    tune_df = pd.DataFrame(train_results)
    name = f"{PROP}-{DESCRIPTOR}-tune-results.csv"
    tune_df.to_csv(name, index=False)
