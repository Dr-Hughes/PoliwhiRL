# -*- coding: utf-8 -*-
from .prediction_tasks import (
    NextFramePredictor,
    RewardPredictor,
    DonePredictor,
    InverseDynamicsModel,
    ForwardDynamicsModel,
    StateContrastiveModel,
    AuxiliaryTaskManager,
)

__all__ = [
    "NextFramePredictor",
    "RewardPredictor",
    "DonePredictor",
    "InverseDynamicsModel",
    "ForwardDynamicsModel",
    "StateContrastiveModel",
    "AuxiliaryTaskManager",
]
