"""CM4AI drug-perturbation analyses (Fig. 4a-c, Extended Data Fig. 5)."""

from .. import config

# One row per cell crop: id, protein (antibody), gene_name, treatment, intensity quantiles.
CM4AI_METADATA = config.ROOTS["CM4AI_DATA"] / "full_subcell_crops_all_z_filtered.csv"
