# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("heretic-llm")
except PackageNotFoundError:
    __version__ = "2.0.0"

from .modifier import Modifier
from .plugin import Context, Plugin
from .scorer import Score, Scorer

__all__ = [
    "Context",
    "Modifier",
    "Plugin",
    "Score",
    "Scorer",
    "__version__",
]
