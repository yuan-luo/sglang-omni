# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for fs_api moved to playground.fs_api."""

from playground.fs_api import create_fs_app, main

__all__ = ["create_fs_app", "main"]


if __name__ == "__main__":
    main()
