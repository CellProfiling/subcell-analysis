"""Compartment and process labels of the proteins in Fig. 6c,e,f, and how the shipped tables were made.

Three tables in ``annotations/`` hold the labels exactly as used for the figures, and
the Fig. 6 analyses read them by default:

``config.U2OS_LOCATIONS``       ``uniprot``, ``locations``: the HPA location annotations
                                of the U2OS cells of each UniProt entry,
                                comma-separated. :func:`read_locations` maps them to
                                Grouping 1 (Supplementary Table 2); an annotation
                                without a Grouping 1 category becomes
                                ``location_not_included_g1``, or stays as it is with
                                ``keep_unmapped``.
``config.SUBORGANELLAR_GOCC``   ``go_cc_term``, ``uniprot``: the members of the 11
                                GO:CC compartments plotted, from a g:Profiler query of
                                14 March 2026 over the entries in
                                ``config.GPROFILER_QUERY_IDS``.
``config.GOBP_TERMS``           ``go_bp_term``, ``go_id``, ``uniprot``: the members of the
                                158 GO:BP processes scored, from a g:Profiler query of
                                March 2026 over the same entries.

This module rebuilds each:

    python -m subcell.multimodal.compartment_labels locations [--out FILE]
    python -m subcell.multimodal.compartment_labels gocc [--ids FILE] [--out FILE]
    python -m subcell.multimodal.compartment_labels gobp --obo go-basic.obo [--ids FILE] [--out FILE]

``locations`` pools the location annotations of the U2OS cells of each gene (the
cell metadata of ``config.features(MODEL)``) and regroups genes by reviewed UniProt
entry as the per-gene image embedding does (``config.ENSG2UNIPROT``). The shipped table
paired genes with entries by matching each gene's mean embedding to the per-entry
mean embedding instead; where several genes share an entry it kept the annotations
of one of them, so a rebuilt table can differ for those entries.

``gocc`` and ``gobp`` repeat the g:Profiler queries (``pip install gprofiler-official``),
which send the UniProt entries to the g:Profiler web service. g:Profiler serves only
its current GO release, so a new query returns today's memberships rather than those
of the shipped tables. ``gobp`` keeps the processes the published set was chosen by
(:func:`select_gobp`): g:Profiler enrichment p < 0.001 over the queried entries, 5 or
more of them as members, no other selected process below it in the GO hierarchy
(``is_a`` in ``--obo``, ``go-basic.obo`` from https://purl.obolibrary.org/obo/go/go-basic.obo),
and none specific to a cell type (``CELL_TYPE_WORDS``) or to viral or stem-cell biology.
"""

from __future__ import annotations

import argparse

import pandas as pd

from subcell import config
from subcell.io.features import create_cell_mapping, generate_uniprot_mapping
from subcell.runlog import log

MODEL = "MAE-CellS-ProtS-Pool"
NOT_INCLUDED = "location_not_included_g1"

# GO:BP processes left out of Fig. 6f by name: cell-type-specific, then cell-state terms
CELL_TYPE_WORDS = [
    "neuron", "neuronal", "axon", "dendrite", "synapse", "synaptic", "muscle", "cardiac",
    "skeletal muscle", "smooth muscle", "T cell", "B cell", "lymphocyte", "leukocyte",
    "erythrocyte", "platelet", "megakaryocyte", "osteoblast", "osteoclast", "chondrocyte",
    "adipocyte", "epithelial", "endothelial", "hepatocyte", "nephron", "kidney",
    "photoreceptor", "retina", "eye", "sperm", "spermatogenesis", "oocyte", "embryo",
    "embryonic", "fetal", "immune response", "inflammatory", "astrocyte", "oligodendrocyte",
    "glial", "cardiomyocyte", "myoblast", "myocyte",
]
CELL_STATE_WORDS = ["viral", "virus", "stem cell"]

# The GO:CC compartments of Fig. 6c, by organelle
SUBORGANELLAR_TERMS = {
    "Mitochondria": ["mitochondrial outer membrane", "mitochondrial inner membrane",
                     "mitochondrial intermembrane space", "mitochondrial matrix"],
    "ER": ["endoplasmic reticulum membrane", "endoplasmic reticulum lumen"],
    "Vesicle": ["vesicle membrane", "vesicle lumen"],
    "Golgi": ["Golgi membrane", "cis-Golgi network", "trans-Golgi network"],
}


def read_locations(path=None, keep_unmapped=False) -> dict[str, list[str]]:
    """UniProt entry -> Grouping 1 locations, from a table written like ``config.U2OS_LOCATIONS``.

    With ``keep_unmapped``, an annotation without a Grouping 1 category is kept as it
    is rather than becoming ``NOT_INCLUDED``.
    """
    df = pd.read_csv(path or config.U2OS_LOCATIONS, sep="\t", dtype=str)
    to_group = grouping1()
    out = {}
    for u, locs in zip(df["uniprot"], df["locations"]):
        groups = (to_group.get(a, NOT_INCLUDED) for a in locs.split(","))
        if keep_unmapped:
            groups = (g if g != NOT_INCLUDED else a for g, a in zip(groups, locs.split(",")))
        out[u] = list(dict.fromkeys(groups))
    return out


def read_gocc(path=None) -> dict[str, set[str]]:
    """GO:CC term -> member UniProt entries, from a table written like ``config.SUBORGANELLAR_GOCC``."""
    df = pd.read_csv(path or config.SUBORGANELLAR_GOCC, sep="\t", dtype=str)
    return {t: set(g["uniprot"]) for t, g in df.groupby("go_cc_term", sort=False)}


def read_gobp(path=None) -> dict[str, set[str]]:
    """GO:BP process -> member UniProt entries, from a table written like ``config.GOBP_TERMS``."""
    df = pd.read_csv(path or config.GOBP_TERMS, sep="\t", dtype=str)
    return {t: set(g["uniprot"]) for t, g in df.groupby("go_bp_term", sort=False)}


def grouping1() -> dict[str, str]:
    """HPA location annotation -> Grouping 1 category (``NOT_INCLUDED`` where it has none)."""
    table = pd.read_csv(config.LOCATION_GROUPS, sep="\t", dtype=str)
    return {o: (g.strip() if isinstance(g, str) and g.strip() else NOT_INCLUDED)
            for o, g in zip(table["Original annotation"], table["Grouping 1"])
            if isinstance(o, str)}


def u2os_locations(features_path=None) -> pd.DataFrame:
    """Rebuild the ``config.U2OS_LOCATIONS`` table from the HPA cell metadata."""
    import torch

    path = features_path or config.features(MODEL)
    log(f"cell metadata of {path}")
    df, _ = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    df = df[(df["atlas_name"] == "U2OS") & df["ensembl_ids"].notna()
            & df["locations"].notna()].reset_index(drop=True)
    log(f"{len(df):,} U2OS cells with location annotations")
    cell_locations = df["locations"].astype(str).str.split(",")

    rows = []
    for entry, cells in generate_uniprot_mapping(create_cell_mapping(df)).items():
        annotations = sorted({a.strip() for c in cells for a in cell_locations[c]})
        rows.append((entry, ",".join(annotations)))
    return pd.DataFrame(rows, columns=["uniprot", "locations"]).sort_values("uniprot")


def query_gocc(uniprot_ids) -> dict[str, list[str]]:
    """GO:CC term -> member UniProt entries among *uniprot_ids*, from g:Profiler's current release."""
    from gprofiler import GProfiler

    res = GProfiler(return_dataframe=True).profile(
        organism="hsapiens", query=list(uniprot_ids), sources=["GO:CC"],
        user_threshold=1.0, all_results=True, no_evidences=False)
    return {name: sorted(members) for name, members in zip(res["name"], res["intersections"])}


def query_gobp(uniprot_ids) -> pd.DataFrame:
    """GO:BP processes enriched (p < 0.05) among *uniprot_ids*: name, GO id, p-value, members."""
    from gprofiler import GProfiler

    res = GProfiler(return_dataframe=True).profile(
        organism="hsapiens", query=list(uniprot_ids), sources=["GO:BP"],
        user_threshold=0.05, no_evidences=False)
    return res.rename(columns={"name": "go_bp_term", "native": "go_id"})[
        ["go_bp_term", "go_id", "p_value", "intersections"]]


def go_parents(obo_path) -> dict[str, list[str]]:
    """GO id -> its ``is_a`` parents, from an OBO file."""
    parents, current = {}, None
    with open(obo_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                current = None
            elif line.startswith("id:"):
                current = line[3:].strip()
                parents.setdefault(current, [])
            elif line.startswith("is_a:") and current:
                parents[current].append(line[5:].split("!")[0].strip())
    return parents


def select_gobp(terms, obo_path) -> pd.DataFrame:
    """The processes of *terms* (from :func:`query_gobp`) that Fig. 6f scores."""
    terms = terms[terms["intersections"].map(len) >= 5]
    ids, parents = set(terms["go_id"]), go_parents(obo_path)
    has_child = {p for g in ids for p in parents.get(g, []) if p in ids}
    words = [w.lower() for w in CELL_TYPE_WORDS + CELL_STATE_WORDS]
    named = terms["go_bp_term"].str.lower().map(lambda n: any(w in n for w in words))
    return terms[~terms["go_id"].isin(has_child) & (terms["p_value"] < 0.001) & ~named]


def gobp_table(terms) -> pd.DataFrame:
    return pd.DataFrame([(t, g, u) for t, g, members in
                         zip(terms["go_bp_term"], terms["go_id"], terms["intersections"])
                         for u in sorted(set(members))],
                        columns=["go_bp_term", "go_id", "uniprot"])


def suborganellar_table(term_members) -> pd.DataFrame:
    terms = [t for group in SUBORGANELLAR_TERMS.values() for t in group]
    return pd.DataFrame([(t, u) for t in terms for u in sorted(term_members.get(t, []))],
                        columns=["go_cc_term", "uniprot"])


def main(which, out=None, ids=None, features_path=None, obo=None):
    if which == "locations":
        table = u2os_locations(features_path)
    else:
        with open(ids or config.GPROFILER_QUERY_IDS) as f:
            uniprot_ids = [line.strip() for line in f if line.strip()]
        log(f"querying g:Profiler with {len(uniprot_ids):,} UniProt entries")
        if which == "gocc":
            table = suborganellar_table(query_gocc(uniprot_ids))
        else:
            if obo is None:
                raise SystemExit("gobp needs --obo, a go-basic.obo file")
            table = gobp_table(select_gobp(query_gobp(uniprot_ids), obo))
    out = out or config.ensure(config.COMPARTMENT_ENRICHMENT_DIR) / f"rebuilt_{which}.tsv"
    table.to_csv(out, sep="\t", index=False)
    log(f"wrote {len(table):,} rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("which", choices=["locations", "gocc", "gobp"])
    ap.add_argument("--out", default=None,
                    help="output TSV (default: rebuilt_<which>.tsv in config.COMPARTMENT_ENRICHMENT_DIR)")
    ap.add_argument("--ids", default=None,
                    help="gocc, gobp: UniProt entries to query, one per line "
                         "(default: config.GPROFILER_QUERY_IDS)")
    ap.add_argument("--obo", default=None, help="gobp: the GO ontology, go-basic.obo")
    ap.add_argument("--features", default=None,
                    help="locations: cell feature file whose metadata to read "
                         f"(default: config.features('{MODEL}'))")
    a = ap.parse_args()
    main(a.which, a.out, a.ids, a.features, a.obo)
