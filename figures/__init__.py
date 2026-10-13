"""One module per manuscript figure, named after the figure and panels it draws.

Each module renders cached results written by the analysis modules in
``subcell``; the computation itself lives there. Figures follow the
specification in ``subcell.style`` and are written to ``config.figure_dir``.

    python -m figures.make_all --list       # each figure's inputs and whether they exist
    python -m figures.make_all fig3c ed3    # selected figures
    python -m figures.ed3_harmonization --help
"""
