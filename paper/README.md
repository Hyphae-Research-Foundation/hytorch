# paper/

- `main.tex` / `main.pdf` — *The Missing Medium: A Transactional Residual Stream for Transformer Training*, v1.1 (two-column, 10 pp; reframed after four external reviews, see `notes/REVIEWS-2026-09.md`). Build: `~/.local/bin/tectonic main.tex` (or any LaTeX with the preamble's packages).
- `make_figs.py` — regenerates `figs/*.pdf` and `figs/png/*.png` from `results/`; each figure function names its data file. `../.venv/bin/python make_figs.py [figN_name ...] [png]`.
- `figs/` — committed so the paper builds standalone; `figs/png/` is what the README embeds.
- `notes/` — working material: `REVIEWS-2026-09.md` (review synthesis and the v1.0 → v1.1 changes), `DRAFT.md` (prose master), `OUTLINE.md`, `STATE-OF-THE-ART.md` (related-work research), `EXPLICACION-PARA-HUMANOS.md` (Spanish lay explanation).

Licensed CC BY-SA 4.0.
