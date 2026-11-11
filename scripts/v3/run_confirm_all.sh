#!/usr/bin/env bash
# v3 confirmatory chain (seeds 5-9). Run ONLY after the commit "Freeze pre-registration v3".
# Each step skips work already done, so the chain can be re-run after an interruption.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
W=${WORKERS:-6}
git log --oneline | grep -q "Freeze pre-registration v3" || { echo "pre-registration not frozen; refusing to run"; exit 1; }
for g in confirm_perm confirm_h5 confirm_main; do
  echo "=== $g $(date)"; $PY scripts/v3/run_cells.py --grid $g --workers $W
done
echo "=== size transfer $(date)"
[ -f results/v3/size_transfer_confirm.json ] || $PY scripts/v3/size_transfer.py --seeds 5,6,7,8,9 --pairs 2:4,4:2 --workers $W --out results/v3/size_transfer_confirm.json
for g in rerun_partial_h5 rerun_partial_h3; do
  echo "=== $g $(date)"; $PY scripts/v3/run_cells.py --grid $g --workers $W
done
echo "=== done $(date)"
