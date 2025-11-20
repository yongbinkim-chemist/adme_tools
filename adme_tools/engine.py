import pandas as pd

import torch

from torchmetrics.regression import R2Score

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from tqdm.auto import tqdm

from typing import Dict, Tuple, List

from .utils import save_model

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: str = "cpu") -> float: 
    
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    return train_loss

def valid_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: str = "cpu") -> Tuple[float, float]:
    
    valid_loss = 0
    r2_metric = R2Score().to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.item()
            r2_metric.update(y_pred, y)
    valid_loss /= len(dataloader)
    valid_r2 = r2_metric.compute().item()
    r2_metric.reset()
    return valid_loss, valid_r2

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str = "cpu") -> Tuple[float, float]:
    
    all_preds = []
    ground_truth = []
    test_loss = 0
    r2_metric = R2Score().to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            ground_truth.append(y)
            all_preds.append(y_pred)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            r2_metric.update(y_pred, y)
    test_loss /= len(dataloader)
    test_r2 = r2_metric.compute().item()
    r2_metric.reset()
    
    print("###### Test Step ######")
    print(f"Test Loss: {test_loss:.3f} | Test R2: {test_r2:.2f}")
    all_preds = torch.cat(all_preds).cpu().reshape(-1)
    ground_truth = torch.cat(ground_truth).cpu().reshape(-1)
    results = {"test_loss": [test_loss], "test_r2": [test_r2]}
    return results, all_preds, ground_truth

def maccs_train(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module = torch.nn.MSELoss(),
                epochs: int = 5,
                device: str = "cpu",
                save: bool = True,
                tune: bool = False) -> Dict[str, List]:

    results = {"epoch": [], "train_loss": [], "valid_loss": [], "valid_r2": []}
    
    # for epoch in tqdm(range(epochs)):
    valid_loss, valid_r2 = None, None
    for epoch in range(epochs):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)
        if not tune:    
            valid_loss, valid_r2 = valid_step(model=model,
                                              dataloader=valid_dataloader,
                                              loss_fn=loss_fn,
                                              device=device)
            # if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f} | Valid R2: {valid_r2:.2f}")
        else:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f}")

        results['epoch'].append(epoch)
        results['train_loss'].append(train_loss)
        results['valid_loss'].append(valid_loss)
        results['valid_r2'].append(valid_r2)
    
    if save:
        if not tune:
            save_model(model=model,
                       target_dir='models',
                       model_name='maccs.pth')
        else:
            save_model(model=model,
                       target_dir='models',
                       model_name='maccs-tuned.pth')
    return results

def mpnn_train(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               valid_dataloader: torch.utils.data.DataLoader,
               optimizer: None,
               loss_fn: None,
               epochs: int = 5,
               device: str = "cpu",
               save: bool = True,
               tune: bool = False):
    
    # accel = "cpu" if device == "cpu" else "auto"
    accel = "cpu" # fix later
    csv_logger = CSVLogger("logs", name="caco2_mpnn")

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        "models",  # Directory where model checkpoints will be saved
        "mpnn",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=save,  # Always save the most recent checkpoint, even if it's not the best
    )

    trainer = pl.Trainer(
        logger=csv_logger,
        enable_checkpointing=save, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator=accel,
        devices=1,
        max_epochs=epochs, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(model, train_dataloader, valid_dataloader)

    log_path = "logs/caco2_mpnn/version_0/metrics.csv"
    df = pd.read_csv(log_path)

    train_mse = df[["epoch", "train_loss_epoch"]].dropna()
    val_mse = df[["epoch", "val/mse"]].dropna()
    val_r2 = df[["epoch", "val/r2"]].dropna()

    results = {
    "train_loss": train_mse,
    "valid_loss": val_mse,
    "valid_r2": val_r2
    }

    return results
