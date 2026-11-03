# Running the Main Grid on Another PC (e.g. Ryzen 5 7600 + RTX 4060)

## What the hardware does

- **CPU cores do the work.** Mode P is dominated by CPU-bound Python:
  - per-step counterfactual rollouts (about 300 per test episode);
  - removal replays;
  - training small critics.

  Each worker process uses one core, so throughput scales with `--workers`. A Ryzen 5 7600 has 6 cores / 12 threads: use **`--workers 10`**, leaving two threads for the system.
- **The GPU is not used by Mode P.** The harness deliberately builds learners on CPU: the networks are tiny, and GPU launch overhead outweighs the gain. The RTX 4060 matters for Mode T (online training, Step 8). Installing the CUDA build of torch now still makes sense.
- **Expected time.** On the original machine (8 threads, 7 workers) a cell took 10–25 minutes. 120 cells on 10 faster workers: roughly **2.5–4 hours**.
- **Keep the PC awake.** Disable sleep for the duration (Settings → System → Power → Screen and sleep → Never). Sleep suspends the run; it does not lose finished cells.

## 1. Install prerequisites

- Python **3.12** (64-bit) from python.org. Tick "Add python.exe to PATH".
- Git for Windows.
- An up-to-date NVIDIA driver (only needed for later steps).

## 2–3. Get the code and EPyMARL

The project lives in a private GitHub repository. EPyMARL is a git submodule pinned to commit `cbc38c0`, so one recursive clone fetches both. Sign in to GitHub when the browser window opens:

```bash
git clone --recursive https://github.com/ManaswinHK/AFML_project.git course_project
```

Then work from inside `course_project`. If you cloned without `--recursive`, fetch EPyMARL afterwards:

```bash
git submodule update --init
```

## 4. Create the environment

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Check:

```bash
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `2.5.1+cu121 True NVIDIA GeForce RTX 4060`.

## 5. Verify before the long run

```bash
.venv/Scripts/python -m pytest -q -p no:warnings
```

All tests must pass (about 3–4 minutes). Then a one-cell smoke run, about 5 minutes:

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid smoke --workers 1
```

## 6. Run the main grid

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid main --workers 10
```

- Results go to `results/mode_p/main/`, one JSON per finished cell.
- **Resumable.** If the run stops, run the same command again; finished cells are skipped. A cell interrupted mid-way restarts from scratch.
- When it finishes, it prints and writes `results/mode_p/main/summary.md`.

## 7. Analyse

```bash
.venv/Scripts/python scripts/analyse_hypotheses.py results/mode_p/main
```

This writes `results/mode_p/main/hypotheses.md`: the pre-registered tests, as amended in `docs/preregistration_amendment.md`.

## 7b. H3 grid (Step 7: all six methods)

The H3 grid has 30 cells: the overlap and partial-synergy layouts × d ∈ {0, 25, 50} × 5 seeds, scored with VDN, QMIX, COMA, MAPPO, NA²Q and STAS. STAS runs on the GPU (the RTX 4060 is used here); everything else runs on CPU. All workers share the one GPU, so use fewer workers than for the main grid. Smoke-test two cells first:

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid h3_smoke --workers 2 --algs vdn,qmix,coma,mappo,na2q,stas
```

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid h3 --workers 6 --algs vdn,qmix,coma,mappo,na2q,stas
```

Results go to `results/mode_p/h3/`.

## 7c. H5 grid (Step 9: coalition estimator), continued from the laptop

The H5 grid has 30 cells: coalition, overlap and partial × d ∈ {0, 50} × b = 2 × 5 seeds. It is followed by `h5_ref`: 10 coalition cells scoring plain NA²Q and STAS. The pre-registration is `docs/h5_preregistration.md`.

The laptop stopped after 16 H5 cells. To continue here:

1. Update the code: `git pull`, then `git submodule update --init`.
2. Copy the finished cells. Results are git-ignored, so copy the folder `results/mode_p/h5/` from the laptop to the same path here, for example via a USB stick or OneDrive. The runner skips every cell whose JSON is present.
3. Run the rest. STAS uses the GPU; as for H3, keep the worker count modest so the workers do not overload the one GPU:

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid h5 --workers 6 --algs ece2,ece3,ece2_mix,ece3_mix,na2q_mix,stas_mix
```

```bash
.venv/Scripts/python scripts/run_mode_p.py --grid h5_ref --workers 6 --algs na2q,stas
```

If a cell fails with a CUDA or CUBLAS error, too many workers are sharing the GPU. Rerun with fewer workers, or hide the GPU (`set CUDA_VISIBLE_DEVICES=` in cmd) and use `--workers 10`.

Expected cost: `na2q_mix` dominates. It took roughly an hour of a coalition d = 0 cell on the laptop, against 4–7 minutes for each ECE variant. The d = 50 cells are slower still.

4. Analyse:

```bash
.venv/Scripts/python scripts/analyse_h5.py
```

This writes `results/mode_p/h5/h5_analysis.md`.

## 8. Bring results back (optional)

To analyse on the original PC, copy the folder `results/mode_p/main/` to the same path there. It is git-ignored, so copy it directly rather than through git.
