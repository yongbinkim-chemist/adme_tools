from collections.abc import Callable
from dataclasses import dataclass

from typing import Callable, Dict, Any

from .data_setup import maccs_create_dataloader, mpnn_create_dataloader
from .models import FCN, mpnn
from .engine import maccs_train, mpnn_train, test_step

@dataclass
class PipelineConfig:
    create_dataloader: Callable
    model_cls: Callable
    trainer: Callable
    tester: Callable
    loader_params: Dict[str, Any]
    model_params: Dict[str, Any]
    torch_run: bool

def get_pipeline(args) -> PipelineConfig:
    
    DESCRIPTOR = args.descriptor
    BATCH_SIZE = args.batch
    SMILES_COL = args.smiles_col
    TARGET_COL = args.target_col

    model_params = {}
    loader_params = {"target_col": TARGET_COL,
                     "batch_size": BATCH_SIZE}

    if DESCRIPTOR == 'maccs':
        
        model_params = {"input_shape": args.input_shape,
                        "hidden_units": args.hidden_units,
                        "output_shape": args.output_shape}

        return PipelineConfig(create_dataloader=maccs_create_dataloader,
                              model_cls = FCN,
                              trainer = maccs_train,
                              tester = test_step,
                              loader_params = loader_params,
                              model_params = model_params,
                              torch_run = True)
    else: # mpnn
        loader_params.update({"mol_col": SMILES_COL,
                              "normalize": False,})

        return PipelineConfig(create_dataloader=mpnn_create_dataloader,
                              model_cls = mpnn,
                              trainer = mpnn_train,
                              tester = None,
                              loader_params = loader_params,
                              model_params = model_params,
                              torch_run = False)
