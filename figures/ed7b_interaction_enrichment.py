"""Extended Data Fig. 7b and Fig. 6d -- enrichment of interacting protein pairs.

Reads the enrichments written by ``subcell.multimodal.interaction_enrichment`` for
CORUM, hu.MAP, STRING and BioPlex under ``config.MULTIMODAL_ENRICHMENT_DIR``. For
each cell line, the hyperparameter configuration (harmonization) with the highest
mean enrichment is kept for every (database, image model, embedding) and bars show
its mean +- s.d. over the 1,000 resamplings.

* ``ed7b_<cell line>``: one panel per image model, bars per embedding: sequence,
  image, MIRAGE co-embedding, MuSIC co-embedding (embeddings with no results are
  left out).
* ``fig6d_<cell line>``: the same for SubCell alone, without MIRAGE, labelled
  Sequence, Image and Multimodal in the colours of the other Fig. 6 panels.

Also writes ``figures/<cell line>/best_results.csv`` (mean +- s.d. of the kept
configurations) beside the results.

    python -m figures.ed7b_interaction_enrichment [--folder DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from subcell import config, naming, style
from subcell.runlog import log
from subcell.multimodal.embeddings import SPACE_COLORS, SPACE_LABELS, SPACES
from subcell.multimodal.interaction_enrichment import evaluation_table
from subcell.style import layout as L

DATABASE_DIRS = {
    "bioplex_level_0": "BioPlex",
    "corum_level_0": "CORUM",
    "humap_level_0": "Hu.MAP",
    "string_level_0": "STRING",
}
ED7B_MODELS = ["DINO4Cells-HPA", "bestfitting", "ViT-ProtS-Pool", "MAE-CellS-ProtS-Pool"]
FIG6D_MODEL = "MAE-CellS-ProtS-Pool"
RANDOM_BASELINES = tuple(f"Random_{d}" for d in naming.RANDOM_BASELINE_DIMS)

# results column -> legend label, in plotting order
EMBEDDING_LABELS = {
    "Seq_Embed": "Sequence (ESM2)",
    "Image_Embed": "Image",
    "Mirage_Image_Seq_Embed_concat": "Image + sequence (MIRAGE)",
    "Music_Embed": "Image + sequence (MuSIC)",
}


def aggregate_results(exp_folder, table=None):
    """All per-resampling enrichments as one frame (Database, Method, Embedding_Type, ...)."""
    table = evaluation_table() if table is None else table
    first_of_cell_line = ~table.duplicated("Cell_Line")
    frames = []
    for i, row in table.iterrows():
        embeddings = [
            col for col in table.columns if "Embed" in col and pd.notna(row[col])
        ]
        embeddings = [f"{c}_concat" if "Mirage" in c else c for c in embeddings]
        if first_of_cell_line[i]:
            embeddings += list(RANDOM_BASELINES)

        for directory, database in DATABASE_DIRS.items():
            for emb_name in embeddings:
                path = f"{exp_folder}/{row['Method_Name']}/{directory}/{emb_name}_results.csv"
                try:
                    df = pd.read_csv(path)
                except FileNotFoundError:
                    log(f"missing: {path}")
                    continue
                df = df.rename(columns={"Embedding": "Embedding_Type", "Result": "Enrichment"})
                random_baseline = emb_name in RANDOM_BASELINES
                df["Embedding_Type"] = "Random" if random_baseline else emb_name
                df["Method"] = "Random" if random_baseline else row["Method"]
                df["Cell_Line"] = row["Cell_Line"]
                df["Harmonized"] = "none" if random_baseline else row["Harmonized"]
                df["Database"] = database
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no enrichment results under {exp_folder}")
    return pd.concat(frames, ignore_index=True)


def best_configurations(cell_line_df):
    """Rows of the harmonization with the highest mean enrichment per (database, model, embedding)."""
    key = ["Database", "Method", "Embedding_Type"]
    group = key + ["Cell_Line", "Harmonized"]
    mean_df = cell_line_df.groupby(group)["Enrichment"].mean().reset_index()
    best = mean_df.loc[mean_df.groupby(key)["Enrichment"].idxmax(), group]
    return best.merge(cell_line_df, on=group, how="inner").reset_index(drop=True)


def summary_table(best_df):
    """``mean ± s.d.`` of the enrichment of each kept configuration."""
    group = ["Database", "Method", "Embedding_Type", "Cell_Line", "Harmonized"]
    stats = best_df.groupby(group)["Enrichment"].agg(["mean", "std"]).reset_index()
    stats["Enrichment"] = stats.apply(lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1)
    return stats.drop(columns=["mean", "std"])


EMBEDDING_COLORS = dict(zip(EMBEDDING_LABELS, style.palettes.categorical(len(EMBEDDING_LABELS))))

# Fig. 6d names and colours its embeddings as the other Fig. 6 panels do, unmuted
FIG6D_LABELS = {col: SPACE_LABELS[s] for s, col in SPACES.items()}
FIG6D_COLORS = {col: SPACE_COLORS[s] for s, col in SPACES.items()}


def _bars(ax, data, method, embeddings, labels=EMBEDDING_LABELS, colors=EMBEDDING_COLORS,
          saturation=0.75):
    data = data.assign(Embedding=data["Embedding_Type"].map(labels))
    sns.barplot(
        data=data, x="Database", y="Enrichment", hue="Embedding",
        order=sorted(DATABASE_DIRS.values()),
        hue_order=[labels[e] for e in embeddings],
        palette={labels[e]: colors[e] for e in embeddings},
        errorbar="sd", ax=ax, linewidth=style.LW_MIN, err_kws={"linewidth": style.LW_AXIS},
        saturation=saturation,
    )
    ax.set_title(naming.display(method))
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend_.remove()
    L.despine(ax)


def _panels(name, best_df, methods, embeddings, width, panel_w, out_dir, height=54,
            labels=EMBEDDING_LABELS, colors=EMBEDDING_COLORS, saturation=0.75):
    """One bar panel per image model, a shared legend below."""
    # The sequence embedding is evaluated on the first row of each cell line only and
    # does not depend on the image model, so the same bars appear in every panel.
    seq = best_df[best_df["Embedding_Type"] == "Seq_Embed"]
    fig = L.figure(width, height=height)
    gap = (width - 14 - len(methods) * panel_w) / (len(methods) - 1) if len(methods) > 1 else 0
    for i, method in enumerate(methods):
        own = best_df[(best_df["Method"] == method) & best_df["Embedding_Type"].isin(embeddings)]
        ax = L.axes(fig, x=12 + i * (panel_w + gap), y=6, w=panel_w, h=height - 24)
        _bars(ax, pd.concat([own, seq.assign(Method=method)]), method, embeddings, labels, colors,
              saturation)
        ax.set_ylabel("Enrichment (Cliff's delta)" if i == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[e]) for e in embeddings]
    fig.legend(handles, [labels[e] for e in embeddings], loc="lower center",
               ncol=len(embeddings), frameon=False, bbox_to_anchor=(0.5, 0.0))
    style.save(fig, name, out_dir)


def main(exp_folder=None, out_dir=None):
    exp_folder = exp_folder or config.MULTIMODAL_ENRICHMENT_DIR
    all_df = aggregate_results(exp_folder)

    style.use_agg()
    with style.nature():
        for cell_line in all_df["Cell_Line"].unique():
            cell_line_df = all_df[all_df["Cell_Line"] == cell_line].reset_index(drop=True)
            best_df = best_configurations(cell_line_df)

            table_dir = config.ensure(f"{exp_folder}/figures/{cell_line}")
            summary_table(best_df).to_csv(f"{table_dir}/best_results.csv", index=False)

            present = [e for e in EMBEDDING_LABELS if e in set(best_df["Embedding_Type"])]
            _panels(f"ed7b_{cell_line}", best_df, ED7B_MODELS, present, L.ED_WIDTH, 32,
                    out_dir or config.figure_dir("ed7"))
            _panels(f"fig6d_{cell_line}", best_df, [FIG6D_MODEL],
                    [e for e in present if "Mirage" not in e], style.WIDTH_1COL, 60,
                    out_dir or config.figure_dir("fig6"),
                    labels=FIG6D_LABELS, colors=FIG6D_COLORS, saturation=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=None, help="enrichment results (default: config.MULTIMODAL_ENRICHMENT_DIR)")
    ap.add_argument("--out-dir", default=None, help="write all figures here (default: figure_dir('ed7') and ('fig6'))")
    a = ap.parse_args()
    main(a.folder, a.out_dir)
