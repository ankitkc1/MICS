"""Local-training utilities."""

from .losses import DiceCrossEntropyLoss, soft_dice_loss

__all__ = ["DiceCrossEntropyLoss", "soft_dice_loss"]
