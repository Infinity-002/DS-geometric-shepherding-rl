# IEEE Paper: Geometric-Informed Recurrent PPO for Shepherding

This folder contains an IEEE-format manuscript written with the official Typst [`charged-ieee`](https://typst.app/universe/package/charged-ieee/) template.

## Files

| Path | Role |
|---|---|
| `main.typ` | Manuscript source |
| `refs.bib` | IEEE-style bibliography |
| `figures/` | Paper figures (copied or extracted from `images/` at the repo root) |
| `fonts/` | TeX Gyre Termes (required by `charged-ieee`) |
| `compile.sh` | One-shot Typst build |
| `geometric-shepherding-rl.pdf` | Compiled paper (generated) |

Author names in `main.typ` are placeholders.

## Build

Typst 0.12+ is required (`charged-ieee` 0.1.4). On this machine the CLI was installed to `~/.local/bin/typst` (v0.15.1).

```bash
cd docs/paper
chmod +x compile.sh
./compile.sh
```

Equivalent:

```bash
typst compile --font-path fonts --root . main.typ geometric-shepherding-rl.pdf
```

The TeX Gyre Termes family lives in `fonts/` so the build does not depend on a system font package.

## Figures

Static plots come from the repo `images/` directory. GIF demos are not embeddable in PDF, so a representative frame was extracted for each rollout.

Rebuild stills after replacing the GIFs:

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("../..")
out = Path("figures")
mapping = {
    "images/ppo_herding_v2_demo.gif": "ppo_v2_still.png",
    "images/v3_structured_3d.gif": "v3_structured_still.png",
    "images/bc_structured_3d.gif": "bc_structured_still.png",
}
for src, dst in mapping.items():
    im = Image.open(root / src)
    n = getattr(im, "n_frames", 1)
    im.seek(min(n // 3, n - 1))
    im.convert("RGB").save(out / dst)
PY
```
