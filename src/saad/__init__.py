#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saad-disentangling")
except PackageNotFoundError:
    __version__ = "1.1.0"

__status__ = "Production"

from . import disentangle_functions, file_handler, plotting
from .config import load_config
from .runner import run

__all__ = [
    "disentangle_functions",
    "plotting",
    "file_handler",
    "load_config",
    "run",
]
