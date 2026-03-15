"""
pyrustler — pure-Python port of the Rust ``rustler`` crate.

Provides the same ``Sampler`` class used by ``rt.data.RelationalDataset``,
as well as the ``pre`` preprocessing module and ``convert_file`` utility.

Sub-modules
-----------
- ``pyrustler.common``       — shared data structures (Node, Edge, …)
- ``pyrustler.pre``          — dataset preprocessing  (``python -m pyrustler.pre``)
- ``pyrustler.fly``          — on-the-fly context sampler
- ``pyrustler.convert_file`` — pickle → JSON converter
"""

from .fly import Sampler
from .common import Edge, Node, SemType, TableType, TableInfo

__all__ = [
    "Sampler",
    "Edge",
    "Node",
    "SemType",
    "TableType",
    "TableInfo",
]
