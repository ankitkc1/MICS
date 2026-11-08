"""Client-selection contracts and reference methods."""

from .base import ClientSelector, ClientState, SelectionContext, SelectionPlan
from .mics import (
    AvailabilityAwareServiceCalibration,
    MICS_SERVICE_TYPES,
    PURE_COVERAGE,
    TRUNCATED_LINEAR,
    MICSSelector,
    StaticServiceCalibration,
    calibrate_availability_aware_service_targets,
    calibrate_static_service_targets,
    expected_component_service,
    expected_truncated_linear_service,
    realised_component_service,
)
from .random import UniformRandomSelector

__all__ = [
    "AvailabilityAwareServiceCalibration",
    "ClientSelector",
    "ClientState",
    "MICSSelector",
    "MICS_SERVICE_TYPES",
    "PURE_COVERAGE",
    "SelectionContext",
    "SelectionPlan",
    "StaticServiceCalibration",
    "TRUNCATED_LINEAR",
    "UniformRandomSelector",
    "calibrate_availability_aware_service_targets",
    "calibrate_static_service_targets",
    "expected_component_service",
    "expected_truncated_linear_service",
    "realised_component_service",
]
