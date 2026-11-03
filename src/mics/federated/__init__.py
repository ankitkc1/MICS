"""Federated utilities that remain independent of a selection policy."""

from .aggregation import ComponentUpdate, aggregate_components

__all__ = ["ComponentUpdate", "aggregate_components"]

