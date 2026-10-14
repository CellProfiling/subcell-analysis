"""How much more similar proteins of one compartment are to each other than to the rest (Fig. 6c).

For a compartment, the similarities of pairs within it (both proteins members) are
compared with those of pairs across its boundary (one member, one not) by Cliff's
delta: the fraction of (within, across) pair combinations in which the within pair
is the more similar, minus the fraction in which it is the less. 1 means every
within pair is more similar than every across pair, 0 that the two overlap
completely. Similarity is the cosine of the mean-centred embeddings, over all
proteins of the MuSIC run (``subcell.multimodal.embeddings``) in each of the three
spaces; proteins without a label count as non-members.

Each delta is estimated from bootstrap draws of equal numbers of within and across
pairs, drawn without replacement when the compartment has that many pairs and with
replacement otherwise; the summary is the mean, s.d. and 2.5-97.5 percentile range
over the draws. Two label sets (``subcell.multimodal.compartment_labels``):

* ``coarse``: the Grouping 1 locations of U2OS proteins (``config.U2OS_LOCATIONS``),
  compartments of 10 or more proteins, 5,000 pairs of each kind x 500 draws.
* ``suborganellar``: the 11 GO:CC compartments (``config.SUBORGANELLAR_GOCC``),
  10,000 pairs x 1,000 draws.

    python -m subcell.multimodal.compartment_enrichment [coarse] [suborganellar]
        [--method-name NAME] [--run-dir DIR] [--locations FILE] [--gocc FILE] [--out-dir DIR]

Outputs, in ``config.COMPARTMENT_ENRICHMENT_DIR``/<method name>/:
``<label set>_summary.csv``, one row per (embedding, compartment).
``figures.fig6c_compartment_enrichment`` plots them.

The published run drew pairs with per-pair Python loops, from a random pool of 10
million across pairs where there were more, and listed proteins in set order, so
its draws cannot be repeated exactly; its estimates and these agree to the
bootstrap error.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config
from subcell.multimodal.compartment_labels import NOT_INCLUDED, read_gocc, read_locations
from subcell.multimodal.embeddings import (METHOD_NAME, SPACES, centered_cosine, input_paths,
                                           read_embedding)
from subcell.runlog import log

# label set -> (pairs of each kind per draw, draws, minimum members)
SETTINGS = {"coarse": (5000, 500, 10), "suborganellar": (10000, 1000, 10)}
SEED = 42


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b) over all (a, b) combinations, exactly, by sorting *b*."""
    b = np.sort(b)
    below = np.searchsorted(b, a, side="left").sum()          # b < a
    above = (len(b) - np.searchsorted(b, a, side="right")).sum()  # b > a
    return (below - above) / (len(a) * len(b))


def draw(rng, n_pairs, n):
    return rng.choice(n_pairs, size=n, replace=n_pairs < n)


def bootstrap(sim, members, n, n_draws, seed=SEED) -> np.ndarray:
    """Cliff's delta of within- vs across-compartment similarities, once per draw."""
    rng = np.random.default_rng(seed)
    others = np.setdiff1d(np.arange(len(sim)), members)
    within = sim[np.ix_(members, members)][np.triu_indices(len(members), k=1)]
    n_across = len(members) * len(others)
    deltas = np.empty(n_draws)
    for d in range(n_draws):
        w = within[draw(rng, len(within), n)]
        i, j = np.divmod(draw(rng, n_across, n), len(others))
        deltas[d] = cliffs_delta(w, sim[members[i], others[j]])
    return deltas


def compartments(label_set, proteins, locations=None, gocc=None) -> dict[str, np.ndarray]:
    """Compartment -> row indices of its members among *proteins*, largest first."""
    index = {p: i for i, p in enumerate(proteins)}
    if label_set == "coarse":
        groups = {}
        for p, locs in read_locations(locations).items():
            for loc in locs:
                if p in index and loc != NOT_INCLUDED:
                    groups.setdefault(loc, set()).add(index[p])
    else:
        groups = {t: {index[p] for p in ps if p in index} for t, ps in read_gocc(gocc).items()}
    min_members = SETTINGS[label_set][2]
    groups = {k: np.array(sorted(v)) for k, v in groups.items() if len(v) >= min_members}
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def main(label_sets=tuple(SETTINGS), method_name=METHOD_NAME, run_dir=None, locations=None,
         gocc=None, out_dir=None):
    paths = input_paths(method_name, run_dir)
    tables = {space: read_embedding(p) for space, p in paths.items()}
    proteins = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    out_dir = config.ensure(out_dir or config.COMPARTMENT_ENRICHMENT_DIR / method_name)
    groups = {s: compartments(s, proteins, locations, gocc) for s in label_sets}
    for s, g in groups.items():
        log(f"{s}: {len(g)} compartments over {len(proteins):,} proteins")

    rows = {s: [] for s in label_sets}
    for space in SPACES:
        log(f"similarities in {space}")
        sim = centered_cosine(tables[space].loc[proteins].to_numpy())
        for s in label_sets:
            n, n_draws, _ = SETTINGS[s]
            for name, members in groups[s].items():
                deltas = bootstrap(sim, members, n, n_draws)
                k = len(members)
                rows[s].append(dict(
                    embedding=space, compartment=name, n_proteins=k,
                    n_within_pairs=k * (k - 1) // 2, n_across_pairs=k * (len(proteins) - k),
                    cliffs_delta_mean=deltas.mean(), cliffs_delta_std=deltas.std(),
                    cliffs_delta_ci_lower=np.percentile(deltas, 2.5),
                    cliffs_delta_ci_upper=np.percentile(deltas, 97.5)))
                log(f"  {s:13s} {name:35s} n={k:5d}  delta={deltas.mean():.3f}")
        del sim

    for s, r in rows.items():
        pd.DataFrame(r).to_csv(out_dir / f"{s}_summary.csv", index=False)
        log(f"wrote {out_dir / f'{s}_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("label_sets", nargs="*", metavar="LABEL_SET",
                    help=f"label sets to score, of {list(SETTINGS)} (default: both)")
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--run-dir", default=None,
                    help="read the run's embeddings from this folder instead")
    ap.add_argument("--locations", default=None,
                    help="coarse labels (default: config.U2OS_LOCATIONS)")
    ap.add_argument("--gocc", default=None,
                    help="sub-organellar labels (default: config.SUBORGANELLAR_GOCC)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if set(a.label_sets) - set(SETTINGS):
        ap.error(f"unknown label set; choose from {list(SETTINGS)}")
    main(a.label_sets or tuple(SETTINGS), a.method_name, a.run_dir, a.locations, a.gocc, a.out_dir)
