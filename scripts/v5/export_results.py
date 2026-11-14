"""Publish the raw v3 results losslessly: gzip every result file next to the original (``x.json`` -> ``x.json.gz``).

Skipped: model checkpoints (``models/``), Sacred's copies of the source code (``_sources/``) and raw stdout logs; the
code is in the repository and every reported number is computed from the files kept. ``results/`` is git-ignored, so
the ``.gz`` files are added with ``git add -f``. ``scripts/v5/restore_results.py`` undoes this.

    .venv/bin/python scripts/v5/export_results.py results/v3
"""
from __future__ import annotations

import gzip
import os
import sys

SKIP_DIRS = {"models", "_sources"}
SKIP_FILES = {"stdout.log"}


def export(root: str) -> tuple[int, int, int]:
    n = raw = packed = 0
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if f.endswith(".gz") or f in SKIP_FILES or f.startswith("."):
                continue
            src = os.path.join(d, f)
            dst = src + ".gz"
            data = open(src, "rb").read()
            if not os.path.exists(dst) or gzip.decompress(open(dst, "rb").read()) != data:
                with open(dst, "wb") as fh:
                    fh.write(gzip.compress(data, 9, mtime=0))     # mtime=0: identical bytes on re-export
            n, raw, packed = n + 1, raw + len(data), packed + os.path.getsize(dst)
    return n, raw, packed


if __name__ == "__main__":
    for root in sys.argv[1:]:
        n, raw, packed = export(root)
        print(f"{root}: {n} files, {raw / 1e6:.1f} MB -> {packed / 1e6:.1f} MB gzipped")
