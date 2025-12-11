"""Utility functions for cross-platform support."""

from arc_meshchop.utils.device import (
    DeviceType,
    enable_mps_fallback,
    get_device,
    get_device_info,
)

__all__ = [
    "DeviceType",
    "enable_mps_fallback",
    "get_device",
    "get_device_info",
]
