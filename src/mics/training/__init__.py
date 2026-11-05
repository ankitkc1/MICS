"""Local-training utilities."""

from .local import LocalTrainingConfig, LocalTrainingResult, train_local_client
from .losses import DiceCrossEntropyLoss, soft_dice_loss

__all__ = [
    "DiceCrossEntropyLoss",
    "LocalTrainingConfig",
    "LocalTrainingResult",
    "soft_dice_loss",
    "train_local_client",
]
