"""Function prediction (Fig. 3d, Extended Data Fig. 8c): how well per-gene image
embeddings predict Reactome and GO membership.

An MLP (``subcell.classification.nested_cv``) is cross-validated over 10 folds on
every embedding of ``config.FUNCTION_PREDICTION_MODELS`` (per-gene means of the
cell features, raw and Harmony-corrected), for GO Biological Process, GO Cellular
Component and Reactome at two hierarchy levels each. The whole grid is repeated on
three fold splits, one results tree per repeat (``config.FUNCTION_PREDICTION_REPEATS``).
Random-feature baselines are scored alongside.

    python -m subcell.classification.function_prediction features   # per-gene means + Harmony where missing
    python -m subcell.classification.function_prediction prepare    # fold splits, one set per repeat
    python -m subcell.classification.function_prediction status     # done / pending units per repeat
    python -m subcell.classification.function_prediction run        # pending units on this node's GPUs
    python -m subcell.classification.function_prediction plot       # merge, then figures.fig3d_function_prediction

``scripts/function_prediction.sbatch`` runs the ``run`` stage as a Slurm array.

A unit is one (repeat, method, database, fold). Every stage skips what is
already on disk, so any of them can be rerun after a crash or pre-emption.
Inputs: ``config.features(<model>)`` and the GO / Reactome tables read by
``subcell.io.complexes``. Outputs: ``config.FUNCTION_PREDICTION_DIR``.
"""

import argparse
import functools
import glob
import json
import os
import socket
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.io.complexes import load_protein_complex_data_cls
from subcell.io.features import scale
from subcell.io.metadata import function_prediction_methods
from subcell.naming import RANDOM_BASELINE_DIMS
from subcell.classification.nested_cv import cross_val_mlp, cross_val_mlp_folds


CONFIGS = [
    ("reactome", 0),
    ("reactome", 1),
    ("go_bp", 2),
    ("go_bp", 3),
    ("go_cc", 2),
    ("go_cc", 3),
]
# `load_protein_complex_data_cls` hard-codes IterativeStratification(n_splits=10)
N_FOLDS = 10
# The random-feature baselines are scored once per cell line, on the rows of this
# (model, harmonization); pinned by name so that adding a model does not move them.
RANDOM_BASELINE_ROW = ("DINO4Cells-HPA", "atlas_name")

# Train every dropout rate and seed of an architecture at once
# (subcell.classification.nested_cv.train_models_batched); False runs them one by one.
BATCHED = True

MLP_PARAMETERS = {
    "n_layers": [1, 2, 3],
    "hidden_dim": [128, 256, 512],
    "dropout": np.round(np.linspace(0.0, 0.8, 9), 1),
}



@functools.lru_cache(maxsize=None)
def method_rows():
    """The (model, harmonization) rows to evaluate, indexed by Method_Name."""
    return function_prediction_methods().set_index("Method_Name", drop=False)


def atomic_to_csv(df, path):
    """Write via rename so a killed task never leaves a truncated CSV; a file
    whose content would not change is left untouched."""
    text = df.to_csv(index=False)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, path)


def has_random_baselines(row):
    return (row["Method"], row["Harmonized"]) == RANDOM_BASELINE_ROW


def expected_emb_names(row):
    names = ["Image_Embed"]
    if has_random_baselines(row):
        names += [f"Random_{d}" for d in RANDOM_BASELINE_DIMS]
    return names


def generate_random_features(n_genes, n_dim, seed):
    """Deterministic random baseline so every unit builds the same matrix."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_genes, n_dim, generator=generator)


def load_method_embeddings(row):
    """(sorted genes, {emb_name: tensor aligned to those genes}) for one row."""
    path = row["Image_Embed"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the `features` stage")
    emb_df = pd.read_csv(path, index_col=0)
    genes = sorted(emb_df.index)
    data = {
        "Image_Embed": torch.from_numpy(
            np.ascontiguousarray(emb_df.loc[genes].to_numpy(dtype=np.float32))
        )
    }
    if has_random_baselines(row):
        for n_dim in RANDOM_BASELINE_DIMS:
            data[f"Random_{n_dim}"] = generate_random_features(
                len(genes), n_dim, seed=n_dim
            )
    return genes, data


def split_path(repeat, complex_name, hierarchy_level, cell_line):
    return (
        f"{config.FUNCTION_PREDICTION_REPEATS[repeat]}/{complex_name}_level_{hierarchy_level}"
        f"_cell_line_{cell_line}.csv"
    )


def unit_dir(unit):
    return (
        f"{config.FUNCTION_PREDICTION_REPEATS[unit['repeat']]}/{unit['method_name']}"
        f"/{unit['complex_name']}_level_{unit['hierarchy_level']}"
    )


@functools.lru_cache(maxsize=None)
def merged_folds(path):
    """Folds already collapsed into a *_results.csv by an earlier merge."""
    if not os.path.exists(path):
        return frozenset()
    return frozenset(pd.read_csv(path, usecols=["Fold"])["Fold"].unique().tolist())


def emb_done(base, emb_name, fold):
    return os.path.exists(f"{base}/folds/{emb_name}_fold_{fold}.csv") or (
        fold in merged_folds(f"{base}/{emb_name}_results.csv")
    )


def unit_done(unit):
    base = unit_dir(unit)
    row = method_rows().loc[unit["method_name"]]
    return all(emb_done(base, e, unit["fold"]) for e in expected_emb_names(row))


def all_units(repeats=None, methods=None):
    """Full grid, repeat-major then method-major."""
    units = []
    for repeat in repeats if repeats is not None else range(len(config.FUNCTION_PREDICTION_REPEATS)):
        for method_name, row in method_rows().iterrows():
            if methods and row["Method"] not in methods and method_name not in methods:
                continue
            for complex_name, hierarchy_level in CONFIGS:
                for fold in range(1, N_FOLDS + 1):
                    units.append({
                        "repeat": repeat,
                        "method_name": method_name,
                        "complex_name": complex_name,
                        "hierarchy_level": hierarchy_level,
                        "fold": fold,
                    })
    return units


def pending_units(repeats=None, methods=None):
    return [u for u in all_units(repeats, methods) if not unit_done(u)]


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def features(args):
    """Per-gene mean features, raw and harmonized, for models that lack them."""
    from subcell.feature_analysis import harmonization

    meta_df = None
    for method, harmonizations in config.FUNCTION_PREDICTION_MODELS.items():
        rows = method_rows()[method_rows()["Method"] == method]
        if all(os.path.exists(p) for p in rows["Image_Embed"]):
            print(f"[features] {method}: complete", flush=True)
            continue
        cats = [
            c for c in harmonization.HARMONIZE_CATS if "_".join(c) in harmonizations
        ]
        assert len(cats) == len(harmonizations) - 1, (method, harmonizations)
        if meta_df is None:
            meta_df = harmonization.microscope_table()
        print(f"[features] {method}: building {len(cats)} harmonization(s)", flush=True)
        harmonization.harmonize_method(
            str(config.features(method)), cats, meta_df, config.FUNCTION_PREDICTION_CELL_LINES
        )


def prepare(args):
    """Fold splits for every (repeat, database, cell line), seeded by repeat.

    Built once, before any unit runs: every method of a cell line must be scored
    on the same split for the per-fold comparisons in the figure to be paired.
    The split is drawn over the genes of one row for that cell line; every model
    shares the same cell table, so the gene sets are identical, and `run_group`
    refuses a split with genes its embedding lacks.
    """
    for repeat, save_folder in enumerate(config.FUNCTION_PREDICTION_REPEATS):
        config.ensure(save_folder)
        for cell_line in config.FUNCTION_PREDICTION_CELL_LINES:
            todo = [
                (c, lvl) for c, lvl in CONFIGS
                if not os.path.exists(split_path(repeat, c, lvl, cell_line))
            ]
            if not todo:
                continue
            rows = method_rows()[method_rows()["Cell_Line"] == cell_line]
            row = rows[rows["Image_Embed"].map(os.path.exists)].iloc[0]
            genes, _ = load_method_embeddings(row)
            for complex_name, hierarchy_level in todo:
                np.random.seed(repeat)
                prot_df = load_protein_complex_data_cls(
                    genes, complex_name, hierarchy_level
                )
                atomic_to_csv(
                    prot_df, split_path(repeat, complex_name, hierarchy_level, cell_line)
                )
                print(
                    f"[prepare] repeat {repeat} {complex_name} L{hierarchy_level} "
                    f"{cell_line}: {len(prot_df.columns) - 2} complexes, "
                    f"{len(prot_df)} genes",
                    flush=True,
                )
    print("[prepare] done", flush=True)


def status(args):
    units = all_units(args.repeats, args.methods)
    pending = [u for u in units if not unit_done(u)]
    for repeat, save_folder in enumerate(config.FUNCTION_PREDICTION_REPEATS):
        n = sum(u["repeat"] == repeat for u in units)
        p = [u for u in pending if u["repeat"] == repeat]
        print(f"repeat {repeat} ({save_folder}): {n - len(p)}/{n} units done")
        by_method = pd.Series([u["method_name"] for u in p]).value_counts()
        for method_name, count in by_method.sort_index().items():
            print(f"    {method_name:<60} {count:>3} pending")
    print(f"total pending: {len(pending)}")


def group_key(unit):
    return (unit["repeat"], unit["method_name"], unit["complex_name"],
            unit["hierarchy_level"])


def run_group(units, n_jobs):
    """The folds in `units` (one repeat, method and database), trained together:
    every fold's replicas of an architecture share one stack
    (`cross_val_mlp_folds`), one embedding at a time. A fold already written for
    an embedding is skipped."""
    max_bytes = None
    if os.environ.get("SUBCELL_GPU_MEM_GB") and torch.cuda.is_available():
        # a hard cap on this process's CUDA allocations, for a shared GPU; the
        # stacks are sized to well under it, leaving room for fragmentation and
        # the validation passes
        total = torch.cuda.get_device_properties(0).total_memory
        cap = float(os.environ["SUBCELL_GPU_MEM_GB"]) * 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
        max_bytes = 0.6 * cap
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unit = units[0]
    assert len({group_key(u) for u in units}) == 1, units
    row = method_rows().loc[unit["method_name"]]
    base = unit_dir(unit)
    os.makedirs(f"{base}/folds", exist_ok=True)

    path = split_path(
        unit["repeat"], unit["complex_name"], unit["hierarchy_level"], row["Cell_Line"]
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the `prepare` stage")
    prot_df = pd.read_csv(path)
    ground_truth = torch.from_numpy(prot_df.drop(columns=["Genes", "Fold"]).values)

    genes, data = load_method_embeddings(row)
    gene2idx = {gene: i for i, gene in enumerate(genes)}
    missing = [g for g in prot_df["Genes"] if g not in gene2idx]
    if missing:
        raise ValueError(
            f"{unit['method_name']} lacks {len(missing)} split genes, e.g. {missing[:3]}"
        )
    gene_idxs = [gene2idx[g] for g in prot_df["Genes"]]

    for emb_name in expected_emb_names(row):
        folds = [u["fold"] for u in units if not emb_done(base, emb_name, u["fold"])]
        if not folds:
            continue
        print(
            f"repeat {unit['repeat']} | {unit['method_name']} | {unit['complex_name']} "
            f"L{unit['hierarchy_level']} | {emb_name} | folds {folds}",
            flush=True,
        )
        features, _, _ = scale(data[emb_name][gene_idxs])
        inputs = []
        for fold in folds:
            train_idxs = prot_df.index[prot_df["Fold"] != fold]
            test_idxs = prot_df.index[prot_df["Fold"] == fold]
            inputs.append((
                features[train_idxs].to(device),
                ground_truth[train_idxs].to(device).to(float),
                features[test_idxs].to(device),
                ground_truth[test_idxs].to(device).to(float),
            ))
        if BATCHED:
            # the stack sizing is an estimate; on a small card a wide embedding
            # can still overflow, so halve the budget and retry
            budget = max_bytes
            while True:
                try:
                    frames = cross_val_mlp_folds(
                        inputs, MLP_PARAMETERS, n_jobs=n_jobs, max_bytes=budget
                    )
                    break
                except torch.OutOfMemoryError:
                    if budget is None or budget < 2**28:
                        raise
                    torch.cuda.empty_cache()
                    budget /= 2
                    print(f"  out of memory; retrying with {budget / 2**30:.1f} GB "
                          f"stacks", flush=True)
        else:
            frames = [cross_val_mlp(*f, MLP_PARAMETERS, n_jobs=n_jobs) for f in inputs]
        for fold, fold_results_df in zip(folds, frames):
            fold_results_df["Fold"] = fold
            fold_results_df["Method"] = emb_name
            atomic_to_csv(fold_results_df, f"{base}/folds/{emb_name}_fold_{fold}.csv")


def features_ready(unit):
    return os.path.exists(method_rows().loc[unit["method_name"], "Image_Embed"])


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


# A claim from another node cannot be checked by PID; it counts as held until it
# is this old, which outlasts any unit, so a pre-empted job's claims expire.
REMOTE_CLAIM_TTL = 4 * 3600
HOST = socket.gethostname()


def claim_holder_alive(path):
    try:
        with open(path) as f:
            text = f.read().strip()
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False
    host, _, pid = text.rpartition(":")
    if not pid.isdigit():
        # created but not yet written: held while young, stale once old
        return age < 60
    if host and host != HOST:
        return age < REMOTE_CLAIM_TTL
    return pid_alive(int(pid))


def write_claim(path, pid):
    with open(path, "w") as f:
        f.write(f"{HOST}:{pid}")


def release_claim(path, pid):
    """Remove a claim this runner still holds; one already taken over is left."""
    try:
        with open(path) as f:
            if f.read().strip() != f"{HOST}:{pid}":
                return
        os.remove(path)
    except FileNotFoundError:
        pass


def claim(unit):
    """Mark a unit as taken, so concurrent runners never train it twice.

    The claim file holds `host:pid` of the process training the unit. A claim
    whose process is gone (a killed runner or unit) is taken over; one from
    another node is honoured until REMOTE_CLAIM_TTL. Returns the claim's path, or
    None if it is held.
    """
    folds = f"{unit_dir(unit)}/folds"
    os.makedirs(folds, exist_ok=True)
    path = f"{folds}/.claim_fold_{unit['fold']}"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if claim_holder_alive(path):
                return None
            # a stale claim is moved aside before anyone re-creates it; the
            # rename succeeds for one runner only, and the exclusive create
            # that follows decides the new holder. If what was moved turns out
            # to be live (renewed between the check and the rename), it goes
            # back unless a newer claim already took its place.
            aside = f"{path}.stale.{HOST}.{os.getpid()}"
            try:
                os.rename(path, aside)
            except FileNotFoundError:
                continue
            live = claim_holder_alive(aside)
            if live:
                try:
                    os.link(aside, path)
                except FileExistsError:
                    pass
            os.remove(aside)
            if live:
                return None
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{HOST}:{os.getpid()}")
        return path
    return None


def run_units(units, gpus, n_jobs, gpu_mem_gb, tag):
    """Run units as child processes, one per entry of `gpus` at a time, each on
    that GPU.

    A child takes every pending fold of one (repeat, method, database) that no
    other process has claimed and trains them together (`run_group`), scoring
    with `n_jobs` joblib workers. Groups go in list order, except that one whose
    features do not exist yet is passed over until they do, so a model whose
    `features` stage is still running joins as soon as it can. `gpu_mem_gb`
    caps each child's CUDA memory and sizes its stacks.
    """
    log_dir = config.ensure(config.FUNCTION_PREDICTION_DIR / "logs")
    slots = range(len(gpus))
    logs = [open(f"{log_dir}/{tag}_slot{s}.log", "a") for s in slots]
    queue = {}
    for u in units:
        queue.setdefault(group_key(u), []).append(u)
    running = {}
    started, strikes, retired = {}, {}, set()
    failed = []
    n_done = 0
    start = time.time()
    print(f"{len(units)} unit(s) in {len(queue)} group(s) on GPU slots {gpus}, "
          f"{n_jobs} scoring worker(s) each, CUDA cap {gpu_mem_gb or 'none'} GB; "
          f"logs {log_dir}/{tag}_slot*.log", flush=True)

    while queue or running:
        busy = {v[0] for v in running.values()}
        free = [s for s in slots if s not in busy and s not in retired]
        while free:
            taken = []
            for key in list(queue):
                if not features_ready(queue[key][0]):
                    continue
                group = queue.pop(key)
                for u in group:
                    if unit_done(u):
                        continue
                    path = claim(u)
                    if path:
                        taken.append((u, path))
                if taken:
                    break
            if not taken:
                break
            slot = free.pop(0)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpus[slot]),
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            if gpu_mem_gb:
                env["SUBCELL_GPU_MEM_GB"] = str(gpu_mem_gb)
            proc = subprocess.Popen(
                [sys.executable, "-m", "subcell.classification.function_prediction", "_group",
                 json.dumps([u for u, _ in taken]), "--n_jobs", str(n_jobs)],
                env=env, cwd=config.REPO, stdout=logs[slot], stderr=subprocess.STDOUT,
            )
            for _, path in taken:
                write_claim(path, proc.pid)
            running[proc.pid] = (slot, taken)
            started[proc.pid] = time.time()
        if not running:
            print(f"{sum(map(len, queue.values()))} unit(s) left whose features are "
                  f"missing; run `features`, then `run` again", flush=True)
            break
        pid, code = os.wait()
        if pid not in running:
            continue
        slot, taken = running.pop(pid)
        for _, path in taken:
            release_claim(path, pid)
        u = taken[0][0]
        folds = [t[0]["fold"] for t in taken]
        label = (f"repeat {u['repeat']} {u['method_name']} {u['complex_name']}"
                 f" L{u['hierarchy_level']} folds {folds}")
        if code:
            failed.extend(t[0] for t in taken)
            print(f"FAIL {label} (slot {slot}, GPU {gpus[slot]})", flush=True)
            # a slot whose units keep failing at once (a GPU in exclusive mode,
            # a broken device) is retired rather than left to drain the queue
            quick = time.time() - started[pid] < 120
            strikes[slot] = strikes.get(slot, 0) + 1 if quick else 0
            if strikes[slot] >= 3:
                retired.add(slot)
                print(f"slot {slot} retired after 3 quick failures", flush=True)
                if len(retired) == len(gpus):
                    break
        else:
            strikes[slot] = 0
            n_done += len(taken)
            hours = (time.time() - start) / 3600
            print(f"[{n_done}/{len(units)} {hours:.2f}h] OK {label} n={len(taken)} "
                  f"(GPU {gpus[slot]})", flush=True)

    for log in logs:
        log.close()
    if failed:
        raise SystemExit(f"{len(failed)} unit(s) failed")


def run(args):
    units = pending_units(args.repeats, args.methods)
    if args.limit:
        units = units[:args.limit]
    if not units:
        print("nothing pending", flush=True)
        return
    gpus = args.gpus or (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
    tag = f"run_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    run_units(units, gpus, args.n_jobs, args.gpu_mem_gb, tag)


def merge_repeat(save_folder):
    """Collapse per-fold CSVs into per-method and cross-method summaries."""
    for complex_name, hierarchy_level in CONFIGS:
        summary_rows = []
        for method_name in method_rows().index:
            base = f"{save_folder}/{method_name}/{complex_name}_level_{hierarchy_level}"
            fold_dir = f"{base}/folds"
            if not os.path.isdir(fold_dir):
                continue

            emb_names = sorted({
                os.path.basename(p).rsplit("_fold_", 1)[0]
                for p in glob.glob(f"{fold_dir}/*_fold_*.csv")
            })
            method_best = []
            for emb_name in emb_names:
                fold_paths = sorted(glob.glob(f"{fold_dir}/{emb_name}_fold_*.csv"))
                result_df = pd.concat(
                    [pd.read_csv(p) for p in fold_paths], ignore_index=True
                )
                atomic_to_csv(result_df, f"{base}/{emb_name}_results.csv")

                n_folds_found = result_df["Fold"].nunique()
                if n_folds_found != N_FOLDS:
                    print(
                        f"  [warn] {save_folder.name}/{method_name}/{complex_name}"
                        f"_level_{hierarchy_level}/{emb_name}: "
                        f"{n_folds_found}/{N_FOLDS} folds",
                        flush=True,
                    )

                best_idx = result_df.groupby(["Method", "Fold"])["macro_ap"].idxmax()
                best = result_df.loc[best_idx].copy()
                method_best.append(best)

                summary = best.copy()
                summary["Method_Name"] = method_name
                summary_rows.append(summary)

            if method_best:
                atomic_to_csv(
                    pd.concat(method_best, ignore_index=True), f"{base}/all_results.csv"
                )

        if summary_rows:
            out = f"{save_folder}/summary_{complex_name}_level_{hierarchy_level}.csv"
            atomic_to_csv(pd.concat(summary_rows, ignore_index=True), out)
    print(f"[merge] {save_folder}", flush=True)


def plot(args):
    for save_folder in config.FUNCTION_PREDICTION_REPEATS:
        merge_repeat(save_folder)
    # the plot stage also renders Fig. 3d / Extended Data Fig. 8c
    from figures import fig3d_function_prediction

    fig3d_function_prediction.main()


def main(argv=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_selection(p):
        p.add_argument("--repeats", nargs="+", type=int,
                       help="Restrict to these repeat indices (default: all)")
        p.add_argument("--methods", nargs="+",
                       help="Restrict to these Method or Method_Name values")

    sub.add_parser("features")
    sub.add_parser("prepare")
    add_selection(sub.add_parser("status"))

    p = sub.add_parser("run")
    add_selection(p)
    p.add_argument("--gpus", nargs="+",
                   help="GPU of each concurrent unit, e.g. 0 0 1 1 (default: one "
                        "unit per visible GPU)")
    p.add_argument("--n_jobs", type=int, default=4,
                   help="scoring workers per unit (default 4)")
    p.add_argument("--gpu_mem_gb", type=float,
                   help="cap on each unit's CUDA memory, in GB")
    p.add_argument("--limit", type=int, help="Run at most this many pending units")

    sub.add_parser("plot")

    # internal: one group of units, as spawned by `run`
    p = sub.add_parser("_group")
    p.add_argument("units")
    p.add_argument("--n_jobs", type=int, default=4)

    args = parser.parse_args(argv)
    if args.stage == "_group":
        run_group(json.loads(args.units), args.n_jobs)
        return
    {"features": features, "prepare": prepare, "status": status, "run": run,
     "plot": plot}[args.stage](args)


if __name__ == "__main__":
    main()
