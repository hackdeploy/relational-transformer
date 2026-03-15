"""
Data structures shared between preprocessing (pre.py) and sampling (fly.py).

Pure-Python port of rustler/src/common.rs.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TableType(IntEnum):
    """Mirrors the Rust TableType enum."""

    Db = 0
    Train = 1
    Val = 2
    Test = 3

    def to_label(self) -> str:
        return self.name  # "Db", "Train", "Val", "Test"

    @staticmethod
    def from_label(label: str) -> "TableType":
        return TableType[label]


class SemType(IntEnum):
    """Semantic type for a cell value."""

    Number = 0
    Text = 1
    DateTime = 2
    Boolean = 3


@dataclass
class TableInfo:
    node_idx_offset: int = 0
    num_nodes: int = 0


@dataclass
class Edge:
    node_idx: int = 0
    table_name_idx: int = 0
    table_type: TableType = TableType.Db
    timestamp: Optional[int] = None


@dataclass
class Node:
    is_task_node: bool = False
    node_idx: int = 0
    f2p_nbr_idxs: list[int] = field(default_factory=list)
    f2p_edges: list[Edge] = field(default_factory=list)
    timestamp: Optional[int] = None
    table_name_idx: int = 0
    col_name_idxs: list[int] = field(default_factory=list)
    sem_types: list[SemType] = field(default_factory=list)
    number_values: list[float] = field(default_factory=list)
    text_values: list[int] = field(default_factory=list)
    datetime_values: list[float] = field(default_factory=list)
    boolean_values: list[float] = field(default_factory=list)
    class_value_idx: list[int] = field(default_factory=list)
