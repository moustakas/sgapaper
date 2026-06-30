#!/usr/bin/env python
"""Generate figures for the SGA-2025 paper.

Usage examples
--------------
python code/build-figures.py --sky
python code/build-figures.py --size-mag
python code/build-figures.py --redshifts
python code/build-figures.py --desi-completeness
python code/build-figures.py --sga2025-vs-sga2020
python code/build-figures.py --all

Figures are written to tex/figures/. Catalogs are read via
SGA.SGA.read_sga_sample, which requires $SGA_PUBLIC_DIR to be set.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import vstack

from SGA.SGA import read_sga_sample

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(REPO_DIR, 'tex', 'figures')

REGIONS  = ['dr11-south', 'dr11-north']


def read_catalogs(regions=None):
    """Read and concatenate SGA2025 group-primary catalogs for all regions.

    Adds a REGION_LABEL string column to distinguish north from south.

    Returns
    -------
    cat : astropy.table.Table
        All GROUP_PRIMARY rows from the requested regions.
    """
    if regions is None:
        regions = REGIONS

    parts = []
    for region in regions:
        sample, _ = read_sga_sample(region=region, beta=False, verbose=True)
        sample['REGION_LABEL'] = region
        parts.append(sample)

    return vstack(parts)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def set_style(paper=True):
    """Configure matplotlib/seaborn for paper-quality figures."""
    try:
        import seaborn as sns
        rc = {'font.family': 'serif'}
        context = 'paper' if paper else 'talk'
        sns.set_theme(context=context, style='ticks', rc=rc)
        colors = sns.color_palette('deep', 12)
    except ImportError:
        plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_sky(cat, png='sga2025-sky.png'):
    """Sky distribution of SGA-2025 galaxies in RA/Dec.

    Simple scatter / 2-D hexbin map; one panel per imaging region.
    """
    colors = set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (region, label) in zip(axes, [
        ('dr11-south', 'DR11-South'),
        ('dr11-north', 'DR11-North'),
    ]):
        sub = cat[cat['REGION_LABEL'] == region]
        ra  = sub['RA'].data.copy()
        ra[ra > 300] -= 360  # wrap to [-60, 300] for a cleaner map
        hb = ax.hexbin(ra, sub['DEC'], gridsize=250, bins='log',
                       cmap='viridis', mincnt=1, linewidths=0)
        cb = plt.colorbar(hb, ax=ax, label='log$_{10}$(N / bin)')
        ax.set_xlabel('Right Ascension (deg)')
        ax.set_title(f'SGA-2025 {label} ({len(sub):,} primaries)')

    axes[0].set_ylabel('Declination (deg)')

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_sample(cat, png='sga2025-sample.png'):
    """D26 diameter distribution for both imaging regions."""
    colors = set_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.logspace(np.log10(0.1), np.log10(100), 80)
    for region, label, color in [
        ('dr11-south', 'DR11-South', colors[0]),
        ('dr11-north', 'DR11-North', colors[1]),
    ]:
        sub = cat[cat['REGION_LABEL'] == region]
        good = sub['D26'] > 0
        ax.hist(sub['D26'][good], bins=bins, histtype='step',
                color=color, lw=1.5, label=f'{label} ({good.sum():,})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$D_{26}$ (arcmin)')
    ax.set_ylabel('N')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_size_mag(cat, band='R', png='sga2025-size-mag.png'):
    """Isophotal diameter D26 vs. total magnitude in an optical band."""
    colors = set_style()

    magcol = f'COG_MTOT_{band}'
    if magcol not in cat.colnames:
        print(f'Column {magcol} not found; skipping fig_size_mag')
        return

    good = (cat['D26'] > 0) & np.isfinite(cat[magcol]) & (cat[magcol] > 0)
    sub  = cat[good]

    fig, ax = plt.subplots(figsize=(7, 6))

    hb = ax.hexbin(sub[magcol], np.log10(sub['D26']),
                   gridsize=200, bins='log', cmap='viridis',
                   mincnt=1, linewidths=0)
    plt.colorbar(hb, ax=ax, label='log$_{10}$(N / bin)')

    ax.set_xlabel(rf'$m_{{{band.lower()}}}$ (mag)')
    ax.set_ylabel(r'log$_{10}$ $D_{26}$ (arcmin)')
    ax.set_title(f'SGA-2025  N={good.sum():,}')

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_redshifts(cat, png='sga2025-redshifts.png'):
    """Redshift distribution color-coded by source (LVD / DESI / NED)."""
    colors = set_style()

    hasz = cat['Z_IVAR'] > 0
    print(f'  {hasz.sum():,} / {len(cat):,} primaries have a redshift')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: sky map of redshift sources ---
    ax = axes[0]
    source_map = {'LVD': colors[2], 'DESI': colors[0], 'NED': colors[1]}
    noz = ~hasz
    ra_all = cat['RA'].data.copy()
    ra_all[ra_all > 300] -= 360
    ax.scatter(ra_all[noz], cat['DEC'][noz], s=0.3, c='0.8',
               rasterized=True, label='No z')
    for src, color in source_map.items():
        mask = hasz & np.array([r.strip() == src for r in cat['Z_REF']])
        if mask.sum() == 0:
            continue
        ra = cat['RA'][mask].data.copy()
        ra[ra > 300] -= 360
        ax.scatter(ra, cat['DEC'][mask], s=0.5, c=color,
                   rasterized=True, label=f'{src} ({mask.sum():,})')
    ax.set_xlabel('Right Ascension (deg)')
    ax.set_ylabel('Declination (deg)')
    ax.legend(markerscale=5, fontsize=9)

    # --- right: redshift histogram ---
    ax = axes[1]
    bins = np.linspace(0, 0.15, 120)
    for src, color in source_map.items():
        mask = hasz & np.array([r.strip() == src for r in cat['Z_REF']])
        if mask.sum() == 0:
            continue
        ax.hist(cat['Z'][mask], bins=bins, histtype='stepfilled',
                color=color, alpha=0.6, label=src)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('N')
    ax.legend()

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_desi_completeness(cat, png='sga2025-desi-completeness.png'):
    """DESI DR1 spectroscopic completeness fraction as a function of D26."""
    colors = set_style()

    bins = np.logspace(np.log10(0.1), np.log10(20), 40)
    bin_cen = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7, 5))

    for region, label, color in [
        ('dr11-south', 'DR11-South', colors[0]),
        ('dr11-north', 'DR11-North', colors[1]),
        (None,         'Combined',   colors[3]),
    ]:
        sub = cat if region is None else cat[cat['REGION_LABEL'] == region]
        good = sub['D26'] > 0
        sub  = sub[good]

        has_desi = sub['Z_IVAR_DESI'] > 0
        n_all, _  = np.histogram(sub['D26'], bins=bins)
        n_desi, _ = np.histogram(sub['D26'][has_desi], bins=bins)

        frac = np.where(n_all > 0, n_desi / n_all, np.nan)
        ls   = '-' if region is None else '--'
        ax.plot(bin_cen, frac, color=color, lw=1.5, ls=ls,
                label=f'{label}  ({has_desi.sum():,} / {good.sum():,})')

    ax.set_xscale('log')
    ax.set_xlabel(r'$D_{26}$ (arcmin)')
    ax.set_ylabel('DESI DR1 redshift completeness')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='k', lw=0.5, ls=':')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_sga2025_vs_sga2020(cat, sga2020_path=None, png='sga2025-vs-sga2020.png'):
    """D26 comparison between SGA-2025 (fitted) and SGA-2020 (initial)."""
    colors = set_style()

    # INIT_REF == 'SGA2020' flags objects whose initial diameter came from the 2020 catalog
    in_2020 = np.array([r.strip() == 'SGA2020' for r in cat['INIT_REF']])
    good    = in_2020 & (cat['D26'] > 0) & (cat['DIAM_INIT'] > 0)
    sub     = cat[good]
    print(f'  {good.sum():,} primaries with SGA-2020 initial diameter')

    ratio = np.log10(sub['D26'] / sub['DIAM_INIT'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # scatter
    ax = axes[0]
    lim = (np.log10(0.5), np.log10(30))
    ax.hexbin(np.log10(sub['DIAM_INIT']), np.log10(sub['D26']),
              gridsize=150, bins='log', cmap='viridis', mincnt=1, linewidths=0)
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'log$_{10}$ $D_{26}$ SGA-2020 (arcmin)')
    ax.set_ylabel(r'log$_{10}$ $D_{26}$ SGA-2025 (arcmin)')

    # ratio histogram
    ax = axes[1]
    ax.hist(ratio, bins=100, range=(-1, 1), histtype='stepfilled',
            color=colors[0], alpha=0.7)
    ax.axvline(0, color='r', lw=1, ls='--')
    med = np.median(ratio)
    ax.axvline(med, color='k', lw=1, ls=':',
               label=f'Median = {med:+.3f} dex')
    ax.set_xlabel(r'log$_{10}$($D_{26,2025}$ / $D_{26,2020}$)')
    ax.set_ylabel('N')
    ax.legend()

    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate SGA-2025 paper figures.')
    parser.add_argument('--sky',              action='store_true')
    parser.add_argument('--sample',           action='store_true')
    parser.add_argument('--size-mag',         action='store_true')
    parser.add_argument('--redshifts',        action='store_true')
    parser.add_argument('--desi-completeness', action='store_true')
    parser.add_argument('--sga2025-vs-sga2020', action='store_true')
    parser.add_argument('--all',              action='store_true',
                        help='Run all figures')
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    run_all = args.all or not any([
        args.sky, args.sample, args.size_mag, args.redshifts,
        args.desi_completeness, args.sga2025_vs_sga2020,
    ])

    cat = read_catalogs()
    print(f'Loaded {len(cat):,} group primaries total')

    if args.sky or run_all:
        fig_sky(cat)

    if args.sample or run_all:
        fig_sample(cat)

    if args.size_mag or run_all:
        fig_size_mag(cat)

    if args.redshifts or run_all:
        fig_redshifts(cat)

    if args.desi_completeness or run_all:
        fig_desi_completeness(cat)

    if args.sga2025_vs_sga2020 or run_all:
        fig_sga2025_vs_sga2020(cat)


if __name__ == '__main__':
    main()
