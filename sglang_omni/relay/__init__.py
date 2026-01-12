# SPDX-License-Identifier: Apache-2.0
"""Relay module for inter-stage data transfer.

This module provides various relay implementations for transferring data
between pipeline stages:
- SHMRelay: Shared memory relay for local (same-machine) transfers
- NixlRalay: NIXL-based RDMA relay for distributed transfers
"""

from sglang_omni.relay.base_operations import BaseReadableOperation, BaseReadOperation
from sglang_omni.relay.ralay import Ralay
from sglang_omni.relay.shm_relay import SHMReadableOperation, SHMReadOperation, SHMRelay

__all__ = [
    "BaseReadOperation",
    "BaseReadableOperation",
    "Ralay",
    "SHMRelay",
    "SHMReadableOperation",
    "SHMReadOperation",
]
