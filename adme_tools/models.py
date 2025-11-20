import torch

import chemprop

from typing import List

class FCN(torch.nn.Module):

  def __init__(self,
               input_shape: int,
               hidden_units: List[int],
               output_shape: int=1):
    super().__init__()

    layers = []
    dim_in = input_shape
    for h in hidden_units:
      layers.append(torch.nn.Linear(in_features=dim_in, out_features=h, bias=True))
      layers.append(torch.nn.ReLU())
      dim_in = h
    layers.append(torch.nn.Linear(in_features=dim_in, out_features=output_shape, bias=True))
    self.layers = torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)

def mpnn():
    mp = chemprop.nn.BondMessagePassing() # options are BondMessagePassing() or AtomMessagePassing()
    agg = chemprop.nn.MeanAggregation() # options are MeanAggregation(), SumAggregation(), nn.NormAggregation()
    ffnn = chemprop.nn.RegressionFFN()
    metric_list = [chemprop.nn.metrics.MSE(), chemprop.nn.metrics.R2Score()]
    mpnn = chemprop.models.MPNN(mp, agg=agg, predictor=ffnn, metrics=metric_list)
    return mpnn
