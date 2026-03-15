"""
On-the-fly context sampler — pure-Python port of rustler/src/fly.rs.

Provides the ``Sampler`` class with the same interface expected by
``rt.data.RelationalDataset``:

    sampler = Sampler(dataset_tuples, batch_size, seq_len, ...)
    sampler.len_py()            -> int
    sampler.batch_py(batch_idx) -> list[(name, np.ndarray | int)]
    sampler.shuffle_py(epoch)
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from pathlib import Path

import ml_dtypes
import numpy as np

from .common import Edge, Node, SemType, TableType

MAX_F2P_NBRS = 5


# ---------------------------------------------------------------------------
# Loaded dataset container
# ---------------------------------------------------------------------------

class _Dataset:
    """Container for a single preprocessed dataset loaded into memory."""

    __slots__ = ("nodes", "text_emb", "p2f_adj")

    def __init__(
        self,
        nodes: list[Node],
        text_emb: np.ndarray,
        p2f_adj: list[list[Edge]],
    ):
        self.nodes = nodes          # list[Node]
        self.text_emb = text_emb    # uint16 array  [num_texts, d_text]
        self.p2f_adj = p2f_adj      # list[list[Edge]]


class _Item:
    __slots__ = ("dataset_idx", "node_idx")

    def __init__(self, dataset_idx: int, node_idx: int):
        self.dataset_idx = dataset_idx
        self.node_idx = node_idx


# ---------------------------------------------------------------------------
# Public Sampler class
# ---------------------------------------------------------------------------

class Sampler:
    """
    Drop-in replacement for the Rust ``Sampler`` exposed via PyO3.

    Constructor signature and public methods match the Rust version exactly.
    """

    def __init__(
        self,
        dataset_tuples: list[tuple[str, int, int]],
        batch_size: int,
        seq_len: int,
        rank: int,
        world_size: int,
        max_bfs_width: int,
        embedding_model: str,
        d_text: int,
        seed: int,
        target_columns: list[int],
        columns_to_drop: list[list[int]],
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.max_bfs_width = max_bfs_width
        self.d_text = d_text
        self.seed = seed
        self.epoch: int = 0
        self.target_columns = target_columns
        self.columns_to_drop = columns_to_drop

        self.datasets: list[_Dataset] = []
        self.items: list[_Item] = []

        home = os.environ.get("USERPROFILE", os.environ.get("HOME", "."))

        for i, (db_name, node_idx_offset, num_nodes) in enumerate(dataset_tuples):
            pre_path = Path(home) / "scratch" / "pre" / db_name

            # Load nodes (pickle written by pyrustler.pre)
            t0 = time.time()
            with open(pre_path / "nodes.pkl", "rb") as f:
                nodes: list[Node] = pickle.load(f)
            print(f"  loaded nodes.pkl for {db_name} ({len(nodes):,} nodes) in {time.time()-t0:.2f}s")

            # Load text embeddings — raw bf16 stored as uint16
            text_path = pre_path / f"text_emb_{embedding_model}.bin"
            raw = np.fromfile(str(text_path), dtype=np.uint16)
            text_emb = raw.reshape(-1, d_text)

            # Load p2f adjacency (pickle)
            with open(pre_path / "p2f_adj.pkl", "rb") as f:
                p2f_adj: list[list[Edge]] = pickle.load(f)

            target = target_columns[i]

            dataset = _Dataset(nodes=nodes, text_emb=text_emb, p2f_adj=p2f_adj)
            self.datasets.append(dataset)

            # Build item list — every task-table node that has the target column
            for j in range(node_idx_offset, node_idx_offset + num_nodes):
                node = nodes[j]
                if target in node.col_name_idxs:
                    self.items.append(_Item(dataset_idx=i, node_idx=j))

            print(f"  {db_name}: {len(self.items)} items so far")

    # ---- public methods (matching Rust PyO3 interface) -----------------

    def len_py(self) -> int:
        return self._len()

    def batch_py(self, batch_idx: int) -> list:
        return self._batch(batch_idx)

    def shuffle_py(self, epoch: int) -> None:
        self.epoch = epoch
        # Use epoch + seed; note: Python random uses Mersenne Twister not ChaCha
        rng = random.Random((epoch + self.seed) & 0xFFFFFFFFFFFFFFFF)
        rng.shuffle(self.items)

    # ---- private implementation ----------------------------------------

    def _len(self) -> int:
        n = len(self.items)
        if n == 0:
            return 0
        d = self.batch_size * self.world_size
        return (n + d - 1) // d  # ceil division

    def _batch(self, batch_idx: int) -> list:
        bs = self.batch_size
        sl = self.seq_len
        d = self.d_text
        L = bs * sl

        if not self.items:
            # Empty dataset — return zeroed batch
            true_batch_size = 0
        else:
            true_batch_size = min(
                bs,
                len(self.items)
                - self.rank * bs
                - batch_idx * bs * self.world_size,
            )

        # Allocate flat arrays (same layout as the Rust Vecs struct)
        node_idxs = np.full(L, -1, dtype=np.int32)
        f2p_nbr_idxs = np.full(L * MAX_F2P_NBRS, -1, dtype=np.int32)
        table_name_idxs = np.zeros(L, dtype=np.int32)
        col_name_idxs = np.zeros(L, dtype=np.int32)
        class_value_idxs = np.full(L, -1, dtype=np.int32)
        col_name_values = np.zeros(L * d, dtype=np.uint16)
        sem_types = np.zeros(L, dtype=np.int32)
        # Float fields: accumulate as float32 then convert to bf16/uint16
        number_values_f32 = np.zeros(L, dtype=np.float32)
        text_values = np.zeros(L * d, dtype=np.uint16)
        datetime_values_f32 = np.zeros(L, dtype=np.float32)
        boolean_values_f32 = np.zeros(L, dtype=np.float32)
        masks = np.zeros(L, dtype=np.bool_)
        is_targets = np.zeros(L, dtype=np.bool_)
        is_task_nodes = np.zeros(L, dtype=np.bool_)
        is_padding = np.ones(L, dtype=np.bool_)

        for i in range(bs):
            if not self.items:
                break
            j = batch_idx * bs * self.world_size + self.rank * bs + i
            j = j % len(self.items)  # wrap when bs > true_batch_size
            item = self.items[j]
            offset = i * sl

            self._seq(
                item,
                offset,
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values_f32,
                text_values,
                datetime_values_f32,
                boolean_values_f32,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
            )

        # Convert float32 → bfloat16 → uint16  (matches Rust bf16 output)
        number_values = number_values_f32.astype(ml_dtypes.bfloat16).view(np.uint16)
        datetime_values = datetime_values_f32.astype(ml_dtypes.bfloat16).view(np.uint16)
        boolean_values = boolean_values_f32.astype(ml_dtypes.bfloat16).view(np.uint16)

        return [
            ("node_idxs", node_idxs),
            ("f2p_nbr_idxs", f2p_nbr_idxs),
            ("table_name_idxs", table_name_idxs),
            ("col_name_idxs", col_name_idxs),
            ("class_value_idxs", class_value_idxs),
            ("col_name_values", col_name_values),
            ("sem_types", sem_types),
            ("number_values", number_values),
            ("text_values", text_values),
            ("datetime_values", datetime_values),
            ("boolean_values", boolean_values),
            ("masks", masks),
            ("is_targets", is_targets),
            ("is_task_nodes", is_task_nodes),
            ("is_padding", is_padding),
            ("true_batch_size", true_batch_size),
        ]

    def _seq(
        self,
        item: _Item,
        offset: int,
        node_idxs,
        f2p_nbr_idxs,
        table_name_idxs,
        col_name_idxs,
        class_value_idxs,
        col_name_values,
        sem_types,
        number_values,
        text_values,
        datetime_values,
        boolean_values,
        masks,
        is_targets,
        is_task_nodes,
        is_padding,
    ):
        """Fill one sequence (one sample) into the pre-allocated arrays."""
        dataset = self.datasets[item.dataset_idx]
        target_column = self.target_columns[item.dataset_idx]
        columns_to_drop = self.columns_to_drop[item.dataset_idx]
        seed_node_idx = item.node_idx
        d = self.d_text
        sl = self.seq_len

        num_total_nodes = len(dataset.nodes)
        visited = [False] * num_total_nodes

        # BFS frontier: list of (depth, node_idx)
        f2p_ftr: list[tuple[int, int]] = [(0, seed_node_idx)]
        seed_node = dataset.nodes[seed_node_idx]
        p2f_ftr: list[list[int]] = []

        seq_i = 0
        rng = random.Random(
            (self.epoch + seed_node_idx + self.seed) & 0xFFFFFFFFFFFFFFFF
        )

        while True:
            # ---- pick next node ----
            if f2p_ftr:
                depth, nidx = f2p_ftr.pop()
            else:
                depth_choices = [i for i, lst in enumerate(p2f_ftr) if lst]
                if not depth_choices:
                    return
                depth = depth_choices[0]
                r = rng.randrange(len(p2f_ftr[depth]))
                # swap-remove (matches Rust behaviour)
                last = len(p2f_ftr[depth]) - 1
                p2f_ftr[depth][r], p2f_ftr[depth][last] = (
                    p2f_ftr[depth][last],
                    p2f_ftr[depth][r],
                )
                nidx = p2f_ftr[depth].pop()

            if visited[nidx]:
                continue
            visited[nidx] = True

            node = dataset.nodes[nidx]

            # Enqueue foreign→primary edges
            for edge in node.f2p_edges:
                f2p_ftr.append((depth + 1, edge.node_idx))

            # Parent→foreign edges for this node
            p2f_edges = dataset.p2f_adj[nidx]

            db_p2f_ftr: list[int] = []

            for edge in p2f_edges:
                # Only include edges to task table if seed node is from task table
                if (
                    edge.table_name_idx != seed_node.table_name_idx
                    and edge.table_type != TableType.Db
                ):
                    continue

                # Temporal constraint
                if (
                    edge.timestamp is not None
                    and seed_node.timestamp is not None
                    and edge.timestamp > seed_node.timestamp
                ):
                    continue

                if edge.table_type == TableType.Db:
                    db_p2f_ftr.append(edge.node_idx)
                    continue

                while depth + 1 >= len(p2f_ftr):
                    p2f_ftr.append([])
                p2f_ftr[depth + 1].append(edge.node_idx)

            # Sub-sample db edges if exceeding max_bfs_width
            if len(db_p2f_ftr) > self.max_bfs_width:
                idxs = rng.sample(range(len(db_p2f_ftr)), self.max_bfs_width)
            else:
                idxs = list(range(len(db_p2f_ftr)))

            for idx in idxs:
                while depth + 1 >= len(p2f_ftr):
                    p2f_ftr.append([])
                p2f_ftr[depth + 1].append(db_p2f_ftr[idx])

            # ---- emit cells for this node ----
            num_cells = len(node.col_name_idxs)
            for cell_i in range(num_cells):
                col_idx = node.col_name_idxs[cell_i]

                # Drop column if this is the seed node (or same timestamp)
                if (
                    node.node_idx == seed_node_idx
                    and col_idx in columns_to_drop
                ):
                    continue
                if (
                    node.timestamp == seed_node.timestamp
                    and col_idx in columns_to_drop
                ):
                    continue

                si = offset + seq_i

                node_idxs[si] = node.node_idx

                assert len(node.f2p_nbr_idxs) <= MAX_F2P_NBRS
                for j_idx, f2p_nbr in enumerate(node.f2p_nbr_idxs):
                    f2p_nbr_idxs[si * MAX_F2P_NBRS + j_idx] = f2p_nbr

                table_name_idxs[si] = node.table_name_idx
                col_name_idxs[si] = node.col_name_idxs[cell_i]
                class_value_idxs[si] = node.class_value_idx[cell_i]

                # Column-name text embedding (uint16 = bf16 bytes)
                cnidx = col_name_idxs[si]
                col_name_values[si * d : (si + 1) * d] = dataset.text_emb[cnidx]

                sem_types[si] = int(node.sem_types[cell_i])

                number_values[si] = node.number_values[cell_i]

                text_idx = node.text_values[cell_i]
                text_values[si * d : (si + 1) * d] = dataset.text_emb[text_idx]

                datetime_values[si] = node.datetime_values[cell_i]
                boolean_values[si] = node.boolean_values[cell_i]

                is_tgt = (
                    seed_node_idx == node.node_idx
                    and node.col_name_idxs[cell_i] == target_column
                )
                is_targets[si] = is_tgt
                masks[si] = is_tgt

                is_task_nodes[si] = node.is_task_node or (
                    node.col_name_idxs[cell_i] == target_column
                )
                is_padding[si] = False

                seq_i += 1
                if seq_i >= sl:
                    break

            if seq_i >= sl:
                break


# ---------------------------------------------------------------------------
# CLI (matches the Rust ``fly`` subcommand)
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="pyrustler fly sampler benchmark")
    parser.add_argument("db_name", nargs="?", default="rel-f1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_trials", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading sampler for {args.db_name} …")
    tic = time.time()
    sampler = Sampler(
        dataset_tuples=[(args.db_name, 0, 10)],
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        rank=0,
        world_size=1,
        max_bfs_width=256,
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seed=0,
        target_columns=[-1],
        columns_to_drop=[[]],
    )
    print(f"Sampler loaded in {time.time()-tic:.2f}s")

    rng = random.Random()
    times = []
    for _ in range(args.num_trials):
        t0 = time.time()
        bidx = rng.randrange(sampler._len())
        _ = sampler._batch(bidx)
        elapsed_ms = (time.time() - t0) * 1000
        times.append(elapsed_ms)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"Mean: {mean:.1f} ms,\tStd: {std:.1f} ms")


if __name__ == "__main__":
    _cli()
