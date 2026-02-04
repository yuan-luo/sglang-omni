# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from typing import Any


def import_string(path: str) -> Any:
    """
    Dynamically import a library given its import path in string.

    Args:
        path: The import path of the library to import.
            Example: "my_project.my_module.my_function"

    Returns:
        The imported attribute.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Import path must be a non-empty string")

    module_path, _, attr = path.rpartition(".")

    if not module_path and not attr:
        raise ValueError(f"Invalid import path {path}")

    if module_path == "":
        return importlib.import_module(attr)
    else:
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise ImportError(f"Module {module_path!r} has no attribute {attr!r}") from exc
