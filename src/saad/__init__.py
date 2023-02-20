#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

__status__ = "Production"


from . import disentangle_functions, file_handler, plotting

__all__ = [
    "disentangle_functions",
    "plotting",
    "file_handler",
]
