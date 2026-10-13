"""python -m subcell.microscope_robustness [stage ...]   (default: every stage, in order; see the package docstring)"""

import argparse
import time

from subcell import config
from subcell.microscope_robustness import data, examples, geometry, phases, retrieval, transfer

STAGES = {"pixels": data.build_pixels, "phases": phases.run, "identity": retrieval.run, "transfer": transfer.run,
          "geometry": geometry.run, "examples": examples.run}
CHECKS = {"check_sklearn": lambda: print(transfer.check_sklearn())}


def main(stages=None):
    for stage in stages or list(STAGES):
        t0 = time.time()
        {**STAGES, **CHECKS}[stage]()
        print(f"{stage} done ({time.time() - t0:.0f} s) -> {config.MICROSCOPE_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", choices=list(STAGES) + list(CHECKS))
    main(parser.parse_args().stages)
