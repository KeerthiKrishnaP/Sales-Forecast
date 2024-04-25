"""Hyper parameters for ML model"""

from pydantic import BaseModel


class PyTorchNonLinearModel(BaseModel):
    epochs: int
    learning_rate: float
    hidden_neurons: int
