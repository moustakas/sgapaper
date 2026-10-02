# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the companion code and paper repository for the **SGA-2025** (Siena Galaxy Atlas 2025) data release. The SGA-2025 delivers multiwavelength imaging mosaics and ellipse photometry for ~486,000 large, resolved galaxies across ~30,000 deg² of the extragalactic sky, using DESI Legacy Imaging Survey DR11 (optical *grz/griz* + unWISE W1–W4 IR + GALEX FUV/NUV). Full documentation: https://sga.readthedocs.io

The repo will be released publicly alongside the submitted paper and on Zenodo, hence its simple flat structure.

## Directory layout

```
code/   — analysis and figure-generation scripts (Python)
data/   — local copies of catalog files (gitignored *.fits; not committed)
tex/    — LaTeX source; figures go in tex/figures/
```

## Data access

The merged science catalogs are the starting point for all analysis:

```
SGA2025-v1.0.fits              (470,625 galaxies, 1.7 GB; full, deduplicated)
SGA2025-dr11-north-v1.0.fits   ( 90,504 galaxies, 340 MB)
SGA2025-dr11-south-v1.0.fits   (395,435 galaxies, 1.4 GB)
```

Each file has three row-matched extensions: `SGA2025` (science catalog), `ELLIPSEPHOT` (full aperture and curve-of-growth photometry, keyed by `SGAID`), and `TRACTOR` (keyed by `REF_ID`/`SGAID`). The authoritative data model is `/Users/ioannis/code/SGA/doc/sga2025.rst`; re-read it if a column is missing.

**On NERSC:** catalogs live at `$SGA_PUBLIC_DIR/` (`/dvs_ro/cfs/cdirs/cosmo/www/sga/2025`).  
**Locally:** set `SGA_PUBLIC_DIR` to point at a local copy or symlink of those files.

The figure scripts depend on the SGA software stack. Read catalogs via:

```python
from SGA.SGA import read_sga_sample

# Returns (sample, fullsample); sample contains only GROUP_PRIMARY rows
sample, _ = read_sga_sample(region='dr11-south', beta=False, verbose=True)
```

Typical selections:

```python
primaries = cat[cat['GROUP_PRIMARY']]          # one per group (most analyses start here)
lvd       = cat[(cat['SAMPLE'] & 1) != 0]     # Local Volume Database objects
hasz      = cat[cat['Z_IVAR'] > 0]            # has a valid redshift (never use Z==0)
```

## Key catalog columns

| Column | Description |
|---|---|
| `SGAID`, `SGANAME` | Unique integer ID; IAU coordinate name `SGA2025 JXXX.XXX±YY.YYY` |
| `GALAXY`, `ALTNAMES`, `OBJNAME` | Primary & alternate names (from NED cross-IDs); parent-catalog name |
| `RA`, `DEC` | Fitted coordinates (deg) |
| `REGION` | Imaging-region bitmask (`SGA.coadds.REGIONBITS`: dr11-south=1, dr11-north=2) |
| `D26`, `D26_ERR`, `D26_REF` | Isophotal diameter at μ=26 mag/arcsec² (arcmin) + band used |
| `BA`, `PA` | Fitted axis ratio b/a and position angle (deg, N through E) |
| `SMA_MOMENT`, `BA_MOMENT`, `PA_MOMENT` | Moment-based geometry (SMA in arcsec) |
| `SMA50_{G,R,I,Z}` | Optical half-light semi-major axis (arcsec) |
| `FLUX_{band}`, `FLUX_IVAR_{band}` | Nominal flux (nanomaggies; all 10 bands: G R I Z W1–W4 FUV NUV), not corrected for Milky Way extinction |
| `FLUX_REF`, `SMA_FLUX` | Source of the nominal flux (currently `AP04`, i.e., 2×`SMA_MOMENT`) and its aperture (arcsec) |
| `EBV`, `MW_TRANSMISSION_{band}` | Milky Way E(B-V) from SFD98 (mag); transmission (all 10 bands) |
| `Z`, `Z_IVAR`, `Z_REF`, `Z_FLAG` | Adopted redshift + ivar + source (`LVD`/`DESI`/`SDSS`/`NED`) + quality bitmask |
| `Z_PHOT`, `Z_COSMO` (+`_IVAR`) | Photometric and peculiar-velocity-corrected redshifts |
| `DIST`, `DIST_IVAR`, `DIST_REF`, `DIST_METHOD` | Adopted distance (Mpc) + ivar + source (`NED_DIRECT`/`LVD`/`NED`/`ZCOSMO`) + technique |
| `SAMPLE` | Bitmask: 1=LVD, 2=MCLOUDS, 4=GCLPNE, 8=NEARSTAR, 16=INSTAR, 32=OVERLAP |
| `ELLIPSEMODE` | Input fitting flags (1=FIXGEO, 2=RESOLVED, 4=FORCEPSF, ...) |
| `ELLIPSEBIT` | Output fitting/quality flags (e.g., 8=BLENDED, 128=OVERLAP, 16384=FAILGEO) |
| `GROUP_NAME`, `SGAGROUP`, `GROUP_PRIMARY`, `GROUP_MULT` | Group membership |
| `BANDS` | Optical bands available (e.g. `griz`, `grz`) |
| `NSPEC_DESI`, `Z_DESI`, `Z_IVAR_DESI` | DESI DR1 spectroscopy (also `*_SDSS`, `*_NED`, `*_LVD`) |

The `ELLIPSEPHOT` extension (not the `SGA2025` extension) holds:

| Column | Description |
|---|---|
| `COG_MTOT_{band}`, `COG_MTOT_ERR_{band}` | Curve-of-growth total magnitude (all 10 bands) |
| `COG_CHI2_{band}`, `COG_NDOF_{band}` | Curve-of-growth fit quality |
| `SMA50_{band}` | Half-light semi-major axis for W1–W4, FUV, NUV (arcsec) |
| `R{22..26}_{band}` | Isophotal semi-major axis at μ=22–26 mag/arcsec² (arcsec; optical only) |
| `SMA_AP{00–04}`, `FLUX_AP{00–04}_{band}`, `FMASKED_AP{00–04}_{band}` | Aperture photometry at [0.5, 1, 1.25, 1.5, 2]×`SMA_MOMENT` |

Two imaging regions: `dr11-south` (DECam, *griz*) and `dr11-north` (BASS+MzLS, *grz*, Dec ≳ +32°).

Public per-group files live under `https://portal.nersc.gov/project/cosmo/data/sga/2025/data/{region}/{GROUP_NAME[:3]}/{GROUP_NAME}/` (e.g., `SGA2025_{GROUP_NAME}-image.jpg`); web thumbnails under `.../cosmo/sga/2025/html/{region}/{GROUP_NAME[:3]}/{GROUP_NAME}/SGA2025_{GROUP_NAME}-thumb.jpg`.

## Figure script

```bash
# Generate one figure at a time (writes to tex/figures/)
python code/build-figures.py --sky
python code/build-figures.py --size-mag
python code/build-figures.py --redshifts
python code/build-figures.py --redshift-completeness
python code/build-figures.py --sga2025-vs-sga2020
python code/build-figures.py --wxsc100
python code/build-figures.py --overview
python code/build-figures.py --all   # all figures
```

## Building the paper

The paper source is `tex/ms.tex`, submitted to AJ (`\submitjournal{\aj}`).

```bash
cd tex
latexmk -pdf ms.tex       # build PDF
latexmk -pdf -pvc ms.tex  # continuous rebuild on save
latexmk -C                # clean all build artifacts
```

Editorial macros defined in `ms.tex`:
- `\JM{text}` — blue inline note from John
- `\todo{text}` — red bold TODO marker

**Bibliography:** `tex/refs.bib` is managed exclusively by BibDesk. Never edit
it directly. If a citation is missing, tell John what reference is needed and
let him add it in BibDesk.

## Writing style

When drafting or revising prose in `tex/ms.tex`, follow John's published
style (reference: Moustakas et al. 2023, ApJS 269:3):

- **Active voice, "we" as subject** — every sentence. Never use passive
  constructions ("are measured", "is distributed", etc.); convert them to
  "We measure", "We distribute", etc.
- **Long, layered sentences** with em-dashes and semicolons; subordinate
  clauses stack qualifications before the main point.
- **Front-loaded paragraphs** — broad context first, specific detail after.
- **Roadmap in "we" form** — "We organize the remainder of the paper as
  follows. In Section X, we ..."
- **Concrete hedging** — "we suspect", "presumably due to", not "may" or
  "could".

Full style guide: `~/.claude/memory/user_writing_style.md`

## Python dependencies

`numpy`, `matplotlib`, `seaborn`, `astropy`, `healpy`, plus the full SGA software stack (which pulls in `fitsio`, `scipy`, etc.). The SGA environment is documented at https://sga.readthedocs.io and in `/Users/ioannis/code/SGA/CLAUDE.md`.
