"""Where sequence-similar protein pairs fall in the multimodal embedding (Fig. 6e).

Every protein pair of the MuSIC run (``subcell.multimodal.embeddings``) is placed by
the percentile rank of its similarity among all pairs (centred cosine) in the
sequence embedding and in the multimodal co-embedding.

Overview, over all pairs: every pair, thinned to about 30,000; and up to 10,000
pairs each with exactly the same U2OS locations or none in common, among 500,000
random pairs. Zoom, among the pairs in the top 2% by sequence:

* background: every pair, thinned to about 50,000; and up to 5,000 pairs each whose
  two proteins have exactly the same U2OS locations, or none in common, of the
  Grouping 1 locations with unmapped annotations kept (``config.U2OS_LOCATIONS``;
  pairs with an unannotated protein are neither);
* the paralog pairs of ``PARALOGS``, which share their locations or do not.

    python -m subcell.multimodal.paralog_ranks [--method-name NAME] [--run-dir DIR] [--out-dir DIR]

Outputs, in ``config.PARALOG_RANKS_DIR``/<method name>/: ``overview.csv`` and
``background.csv`` (``sequence_pct``, ``multimodal_pct``, ``kind`` in all / same /
none) and ``paralogs.csv`` (one row per pair of ``PARALOGS`` in the run).
``figures.fig6e_paralog_ranks`` plots them.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config
from subcell.multimodal.compartment_labels import read_locations
from subcell.multimodal.embeddings import (METHOD_NAME, input_paths, pair_percentiles,
                                           read_embedding)
from subcell.runlog import log

# Paralog pairs shown in Fig. 6e: (gene 1, UniProt 1, gene 2, UniProt 2, same locations)
PARALOGS = [
    ("RAB1B", "Q9H0U4", "RAB1A", "P62820", True),
    ("SMARCA4", "P51532", "SMARCA2", "P51531", True),
    ("PARP1", "P09874", "PARP2", "Q9UGN5", True),
    ("CDK4", "P11802", "CDK6", "Q00534", True),
    ("CDK4", "P11802", "CDK3", "Q00526", False),
    ("SMARCA4", "P51532", "HELLS", "Q9NRZ9", False),
    ("RAB1B", "Q9H0U4", "RAB8A", "P61006", False),
    ("RPL7", "P18124", "RPL7L1", "Q6DKI1", False),
    ("GPX2", "P18283", "GPX8", "Q8TED1", False),
]

MIN_SEQUENCE_PCT = 98   # the zoom: top 2% by sequence similarity
MIN_MULTIMODAL_PCT = 50 # and the lower edge of the zoom's y axis
N_ALL = 50_000          # zoom background pairs drawn in grey, about
N_CLASS = 5_000         # most zoom background pairs drawn per location class
N_OVERVIEW_ALL = 30_000      # overview pairs drawn in grey, about
N_OVERVIEW_SAMPLE = 500_000  # random pairs classified for the overview
N_OVERVIEW_CLASS = 10_000    # most overview pairs drawn per location class
SEED = 42


def pair_index(i, j, n):
    """Position of pair (i, j), i < j, in ``np.triu_indices(n, k=1)`` order."""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def classify(pairs, rows, cols, locs):
    """Positions among *pairs* whose proteins share every location, and none."""
    same, none = [], []
    for k in pairs:
        a, b = locs[rows[k]], locs[cols[k]]
        if a and b:
            if a == b:
                same.append(k)
            elif not a & b:
                none.append(k)
    return same, none


def frame(seq_pct, mm_pct, layers):
    return pd.concat([pd.DataFrame({"sequence_pct": seq_pct[k], "multimodal_pct": mm_pct[k],
                                    "kind": kind}) for kind, k in layers], ignore_index=True)


def overview(seq_pct, mm_pct, proteins, locations) -> pd.DataFrame:
    n = len(proteins)
    rows, cols = np.triu_indices(n, k=1)
    rng = np.random.default_rng(SEED)
    i, j = (rng.integers(0, n, N_OVERVIEW_SAMPLE) for _ in range(2))
    i, j = np.minimum(i, j)[i != j], np.maximum(i, j)[i != j]
    same, none = classify(pair_index(i, j, n), rows, cols,
                          [frozenset(locations.get(p, ())) for p in proteins])
    same, none = (rng.choice(c, N_OVERVIEW_CLASS, replace=False) if len(c) > N_OVERVIEW_CLASS
                  else np.array(c, dtype=np.int64) for c in (same, none))
    every = np.arange(0, len(seq_pct), max(1, len(seq_pct) // N_OVERVIEW_ALL))
    return frame(seq_pct, mm_pct, [("all", every), ("none", none), ("same", same)])


def background(seq_pct, mm_pct, proteins, locations) -> pd.DataFrame:
    shown = np.flatnonzero(seq_pct >= MIN_SEQUENCE_PCT)
    rows, cols = np.triu_indices(len(proteins), k=1)
    same, none = classify(shown, rows, cols, [frozenset(locations.get(p, ())) for p in proteins])
    np.random.seed(SEED)
    same, none = (np.random.choice(c, N_CLASS, replace=False) if len(c) > N_CLASS else np.array(c)
                  for c in (same, none))
    log(f"{len(shown):,} pairs in the top {100 - MIN_SEQUENCE_PCT}% by sequence; "
        f"drawing {len(same):,} with the same locations, {len(none):,} with none shared")
    every = shown[::max(1, len(shown) // N_ALL)]
    return frame(seq_pct, mm_pct, [("all", every), ("none", none), ("same", same)])


def paralogs(seq_pct, mm_pct, proteins) -> pd.DataFrame:
    index = {p: i for i, p in enumerate(proteins)}
    rows = []
    for g1, u1, g2, u2, same in PARALOGS:
        if u1 not in index or u2 not in index:
            log(f"  {g1}-{g2} not in the run, left out")
            continue
        k = pair_index(*sorted((index[u1], index[u2])), len(proteins))
        rows.append(dict(pair=f"{g1}-{g2}", same_locations=same,
                         sequence_pct=seq_pct[k], multimodal_pct=mm_pct[k]))
        log(f"  {g1}-{g2}: sequence {seq_pct[k]:.1f}, multimodal {mm_pct[k]:.1f}")
    return pd.DataFrame(rows)


def main(method_name=METHOD_NAME, run_dir=None, out_dir=None):
    paths = input_paths(method_name, run_dir)
    tables = {space: read_embedding(paths[space]) for space in ("sequence", "multimodal")}
    proteins = sorted(set(tables["sequence"].index) & set(tables["multimodal"].index))
    log(f"percentile ranks over {len(proteins):,} proteins")
    seq_pct, mm_pct = (pair_percentiles(t.loc[proteins].to_numpy()) for t in tables.values())

    out_dir = config.ensure(out_dir or config.PARALOG_RANKS_DIR / method_name)
    locations = read_locations(keep_unmapped=True)
    overview(seq_pct, mm_pct, proteins, locations).to_csv(out_dir / "overview.csv", index=False)
    background(seq_pct, mm_pct, proteins, locations).to_csv(out_dir / "background.csv", index=False)
    paralogs(seq_pct, mm_pct, proteins).to_csv(out_dir / "paralogs.csv", index=False)
    log(f"wrote {out_dir}/overview.csv, background.csv, paralogs.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--run-dir", default=None,
                    help="read the run's embeddings from this folder instead")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.run_dir, a.out_dir)
