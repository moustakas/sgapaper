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

The merged science catalogs (one FITS file per imaging region) are the starting point for all analysis:

```
SGA2025-dr11-north-v1.0.fits   ( 90,504 galaxies, 340 MB)
SGA2025-dr11-south-v1.0.fits   (395,435 galaxies, 1.4 GB)
```

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
| `GALAXY`, `ALTNAMES` | Primary & alternate human-readable names (from NED CROSSIDS) |
| `RA`, `DEC` | Fitted coordinates (deg) |
| `D26`, `D26_ERR` | Isophotal diameter at μ=26 mag/arcsec² (arcmin) |
| `BA`, `PA` | Fitted axis ratio b/a and position angle (deg, N through E) |
| `Z`, `Z_IVAR`, `Z_REF` | Adopted redshift + ivar + source (`LVD`/`DESI`/`NED`) |
| `DIST`, `DIST_IVAR`, `DIST_REF` | Adopted distance (Mpc) + ivar + source (`LVD`/`NED`) |
| `SAMPLE` | Bitmask: bit 1=LVD, 2=MCLOUDS, 4=GCLPNE, 8=NEARSTAR, 16=INSTAR |
| `ELLIPSEBIT` | Output fitting/quality flags (bit 16384=FAILGEO) |
| `GROUP_NAME`, `SGAGROUP`, `GROUP_PRIMARY`, `GROUP_MULT` | Group membership |
| `BANDS` | Optical bands available (e.g. `griz`, `grz`) |
| `EBV` | Milky Way E(B-V) from SFD98 (mag) |
| `COG_MTOT_{band}` | Curve-of-growth total magnitude (all 10 bands: G R I Z W1–W4 FUV NUV) |
| `SMA50_{band}` | Half-light semi-major axis (arcsec) |
| `R{22..26}_{band}` | Isophotal semi-major axis at μ=22–26 mag/arcsec² (arcsec; optical only) |
| `NSPEC_DESI`, `Z_DESI`, `Z_IVAR_DESI` | DESI DR1 spectroscopy |

Two imaging regions: `dr11-south` (DECam, *griz*) and `dr11-north` (BASS+MzLS, *grz*, Dec ≳ +32°).

## Figure script

```bash
# Generate one figure at a time (writes to tex/figures/)
python code/build-figures.py --sky
python code/build-figures.py --size-mag
python code/build-figures.py --redshifts
python code/build-figures.py --desi-completeness
python code/build-figures.py --sga2025-vs-sga2020
python code/build-figures.py --all   # all figures
```

## Building the paper

```bash
cd tex
latexmk -pdf <paper>.tex       # build PDF
latexmk -pdf -pvc <paper>.tex  # continuous rebuild on save
latexmk -C                     # clean all build artifacts
```

## Python dependencies

`numpy`, `matplotlib`, `seaborn`, `astropy`, plus the full SGA software stack (which pulls in `fitsio`, `scipy`, etc.). The SGA environment is documented at https://sga.readthedocs.io and in `/Users/ioannis/code/SGA/CLAUDE.md`.
