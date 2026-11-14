"""Restore published raw results: decompress every ``x.gz`` under the given roots to ``x`` (skips existing files).

    .venv/bin/python scripts/v5/restore_results.py results/v3
"""
from __future__ import annotations

import gzip
import os
import sys

if __name__ == "__main__":
    n = 0
    for root in sys.argv[1:] or ["results/v3"]:
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith(".gz") and not os.path.exists(os.path.join(d, f[:-3])):
                    with open(os.path.join(d, f[:-3]), "wb") as fh:
                        fh.write(gzip.decompress(open(os.path.join(d, f), "rb").read()))
                    n += 1
    print(f"restored {n} files")
