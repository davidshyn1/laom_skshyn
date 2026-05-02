#!/usr/bin/env python3
"""Merge {environment}-500x-train_*.hdf5 shards into one HDF5.

Shard files are ordered by the numeric suffix in ...train_<n>.hdf5 (typically n starts at 1).
Episode groups in the merged file are renumbered 0..N-1 in that shard order.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np


def shard_sort_key(path: str) -> int:
    m = re.search(r"train_(\d+)\.hdf5$", path, re.I)
    if not m:
        raise ValueError(f"Unexpected shard name (expected ...train_<n>.hdf5): {path}")
    return int(m.group(1))


def _normalize_attr_value(val):
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):
        return val.item()
    return val


def _root_attrs_dict(h5: h5py.File) -> dict:
    return {k: _normalize_attr_value(h5.attrs[k]) for k in h5.attrs.keys()}


def _copy_root_attrs(src: h5py.File, dst: h5py.File) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _root_attrs_differ_excluding_dataset_return(ref: dict, shard: dict) -> list[str]:
    keys = sorted(set(ref.keys()) | set(shard.keys()))
    diffs = []
    for k in keys:
        if k == "dataset_return":
            continue
        if ref.get(k) != shard.get(k):
            diffs.append(k)
    return diffs


def _merged_dataset_return(out: h5py.File, global_idx: int, paths: list[str]) -> float:
    """Mean episode return over the merged file (matches collect_data semantics)."""
    traj_returns: list[float] = []
    for k in sorted(out.keys(), key=lambda x: int(x)):
        g = out[k]
        if "traj_return" in g.attrs:
            traj_returns.append(float(np.asarray(g.attrs["traj_return"]).item()))
    if global_idx > 0 and len(traj_returns) == global_idx:
        return float(np.mean(traj_returns))

    weighted_sum = 0.0
    n_eps = 0
    for fp in paths:
        with h5py.File(fp, "r") as src:
            eps = sorted(int(k) for k in src.keys())
            n = len(eps)
            if "dataset_return" not in src.attrs:
                raise ValueError(
                    f"Cannot infer global mean return: episode groups lack full traj_return "
                    f"({len(traj_returns)}/{global_idx} found) and shard has no dataset_return: {fp}"
                )
            weighted_sum += float(np.asarray(src.attrs["dataset_return"]).item()) * n
            n_eps += n
    if n_eps != global_idx:
        raise RuntimeError(f"Episode count mismatch for weighted return: shards={n_eps}, merged={global_idx}")
    return weighted_sum / n_eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-e",
        "--environment",
        default="hopper-hop",
        choices=["walker-run", "cheetah-run", "hopper-hop", "humanoid-walk"],
        help="Environment name used in shard filename placeholders.",
    )
    ap.add_argument("--data-dir", default="/yj_hdd/skshyn/lam/dataset/data")
    ap.add_argument(
        "--pattern",
        default="{environment}-500x-train_*.hdf5",
        help="Glob under data-dir (should not match test set). Supports {environment}.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="/yj_hdd/skshyn/lam/dataset/data/{environment}-500x-train_merged.hdf5",
        help="Merged output path. Supports {environment}.",
    )
    args = ap.parse_args()

    pattern = args.pattern.format(environment=args.environment)
    output = args.output.format(environment=args.environment)

    paths = glob.glob(os.path.join(args.data_dir, pattern))
    paths = [p for p in paths if "test" not in os.path.basename(p).lower()]
    paths = sorted(paths, key=shard_sort_key)
    if not paths:
        print("No train shards found.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    tmp_out = output + ".partial"

    if os.path.exists(tmp_out):
        os.unlink(tmp_out)

    global_idx = 0
    ref_attrs: dict | None = None
    with h5py.File(tmp_out, "w") as out:
        for fp in paths:
            print(f"Shard {shard_sort_key(fp)}: {fp}", flush=True)
            with h5py.File(fp, "r") as src:
                shard_attrs = _root_attrs_dict(src)
                if ref_attrs is None:
                    ref_attrs = shard_attrs
                    _copy_root_attrs(src, out)
                else:
                    diff_keys = _root_attrs_differ_excluding_dataset_return(ref_attrs, shard_attrs)
                    if diff_keys:
                        raise ValueError(
                            f"Root HDF5 attrs differ from first shard; refusing to merge.\n"
                            f"  first shard: {paths[0]}\n"
                            f"  this shard: {fp}\n"
                            f"  differing keys: {diff_keys}"
                        )
                eps = sorted(int(k) for k in src.keys())
                for local in eps:
                    src.copy(str(local), out, name=str(global_idx))
                    global_idx += 1
                    if global_idx % 500 == 0:
                        print(f"  episodes merged: {global_idx}", flush=True)

        if global_idx > 0:
            merged_dr = _merged_dataset_return(out, global_idx, paths)
            out.attrs["dataset_return"] = merged_dr
            print(f"  dataset_return (global mean over {global_idx} episodes): {merged_dr}", flush=True)

    os.replace(tmp_out, output)
    print(f"Done: {global_idx} episodes -> {output}", flush=True)


if __name__ == "__main__":
    main()
