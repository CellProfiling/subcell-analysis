"""Fig. 4b and Extended Data Fig. 5 -- drug-perturbation signatures in CM4AI.

Renders the UMAP of the SubCell embeddings coloured by treatment (Fig. 4b) and
the intensity and embedding volcano plots for paclitaxel and vorinostat
(Extended Data Fig. 5); the per-protein tables behind Fig. 4c are written along
the way, but the joint intensity-versus-sliced-Wasserstein scatter of Fig. 4c is
not drawn here. See ``subcell.perturbation.cm4ai_signatures``.

Fig. 4a (drug prediction) is computed by ``subcell.perturbation.cm4ai_drug_prediction``.
The JUMP-Pilot analyses (Fig. 4e-h, Extended Data Figs. 6 and 9) are in ``jump/``
(see ``jump/README.md``).

Usage
-----
    python -m figures.fig4_perturbation [--out-dir DIR]
"""

from __future__ import annotations

import argparse

from subcell.perturbation import cm4ai_signatures


def main(out_dir=None):
    cm4ai_signatures.main(out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="write all figures here (default: figure_dir('fig4') and ('ed5'))")
    main(ap.parse_args().out_dir)
