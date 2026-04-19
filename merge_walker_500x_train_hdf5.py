#!/usr/bin/env python3
"""Merge walker-run-500x-train_*.hdf5 shards into one HDF5 (episode groups renumbered 0..N-1)."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py


def shard_sort_key(path: str) -> int:
    m = re.search(r"train_(\d+)\.hdf5$", path, re.I)
    if not m:
        raise ValueError(f"Unexpected shard name (expected ...train_<n>.hdf5): {path}")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/yj_hdd/skshyn/lam/dataset/data")
    ap.add_argument(
        "--pattern",
        default="walker-run-500x-train_*.hdf5",
        help="Glob under data-dir (should not match test set).",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="/yj_hdd/skshyn/lam/dataset/data/walker-run-500x-train_merged.hdf5",
    )
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.data_dir, args.pattern))
    paths = [p for p in paths if "test" not in os.path.basename(p).lower()]
    paths = sorted(paths, key=shard_sort_key)
    if not paths:
        print("No train shards found.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    tmp_out = args.output + ".partial"

    if os.path.exists(tmp_out):
        os.unlink(tmp_out)

    global_idx = 0
    with h5py.File(tmp_out, "w") as out:
        for fp in paths:
            print(f"Shard {shard_sort_key(fp)}: {fp}", flush=True)
            with h5py.File(fp, "r") as src:
                eps = sorted(int(k) for k in src.keys())
                for local in eps:
                    src.copy(str(local), out, name=str(global_idx))
                    global_idx += 1
                    if global_idx % 500 == 0:
                        print(f"  episodes merged: {global_idx}", flush=True)

    os.replace(tmp_out, args.output)
    print(f"Done: {global_idx} episodes -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
