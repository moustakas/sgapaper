#!/usr/bin/env python
"""Exploratory figures for evaluating "default" integrated/total photometry
for the SGA-2025 catalog.

Compares the extrapolated curve-of-growth total magnitude (COG_MTOT) against
fixed-aperture magnitudes (AP00-AP04, 0.5-2.0 x SMA_MOMENT), and checks how
COG fit quality (COG_CHI2/COG_NDOF) and aperture masking (FMASKED_APxx)
track their disagreement, and how it depends on galaxy size (D26). Figures
are exploratory only; useful pieces get promoted into build-figures.py once
the diagnostics settle.

Usage examples
--------------
python code/phot.py --reliability
python code/phot.py --masking
python code/phot.py --aperture-series
python code/phot.py --moment-vs-d26
python code/phot.py --colors
python code/phot.py --all

Figures are written to code/phot-figures/. Catalogs are read via
SGA.SGA.read_sga_sample, which requires $SGA_PUBLIC_DIR to be set.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from SGA.SGA import read_sga_sample
from util import hess_contours, running_binstat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(REPO_DIR, 'code', 'phot-figures')

# band pairs used for the color-color comparison; chosen to avoid i-band,
# which the dr11-north (grz) region lacks
COLOR_PAIRS = [('G', 'R'), ('R', 'Z'), ('R', 'W1'), ('NUV', 'R')]


def read_catalogs(primary_only=False):
    """Read the merged (region-combined) SGA2025 catalog."""
    sample, fullsample = read_sga_sample(beta=False, verbose=True)
    return sample if primary_only else fullsample


def plot_style(font_scale=0.8, paper=False, talk=True):
    """Seaborn plot style wrapper (matches build-figures.py)."""
    import seaborn as sns
    rc = {'font.family': 'serif'}
    palette, context = 'Set2', 'talk'
    if paper:
        context = 'paper'
        palette = 'deep'
        rc.update({'text.usetex': False})
    if talk:
        context = 'talk'
        palette = 'deep'
    sns.set_theme(context=context, style='ticks', font_scale=font_scale, rc=rc)
    sns.set_palette(palette, 12)
    colors = sns.color_palette()
    return sns, colors


def flux2mag(flux):
    """Convert nanomaggies to AB mag; non-positive flux -> NaN."""
    flux = np.asarray(flux, dtype=float)
    mag  = np.full_like(flux, np.nan)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def get_sample(cat, band='R'):
    """Isolated, unflagged, well-resolved galaxies with the requested band.

    Cuts: singleton groups (GROUP_MULT==1), no SAMPLE bits set (excludes
    LVD/MCLOUDS/GCLPNE/NEARSTAR/INSTAR/OVERLAP), 0.5 < D26 < 5 arcmin, and
    `band` present in BANDS (most, but not all, objects have every band).
    """
    good = (
        (cat['GROUP_MULT'] == 1) &
        (cat['SAMPLE'] == 0) &
        (cat['D26'] > 0.5) &
        (cat['D26'] < 5.)
    )
    bands = np.char.strip(np.asarray(cat['BANDS'], dtype=str))
    good &= np.char.find(bands, band.lower()) >= 0

    sub = cat[good]
    print(f'  Sample cut ({band}): {good.sum():,} / {len(cat):,} galaxies pass '
          f'(GROUP_MULT==1, SAMPLE==0, 0.5<D26<5, has {band})')
    return sub


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_reliability(cat, band='R', fmasked_thresh=0.2, png=None):
    """COG_MTOT - m_AP04 vs. COG fit-quality diagnostics.

    Note: COG_DMAG is a parameter of the COG model itself (part of what
    produces COG_MTOT), not an independent check on aperture correctness,
    so it is intentionally excluded here.
    """
    plot_style()
    b = band.upper()
    sub = get_sample(cat, band=b)

    cogmag  = np.asarray(sub[f'COG_MTOT_{b}'], dtype=float)
    apmag   = flux2mag(sub[f'FLUX_AP04_{b}'])
    fmasked = np.asarray(sub[f'FMASKED_AP04_{b}'], dtype=float)
    chi2    = np.asarray(sub[f'COG_CHI2_{b}'], dtype=float)
    ndof    = np.asarray(sub[f'COG_NDOF_{b}'], dtype=float)

    good = (
        np.isfinite(cogmag) & (cogmag > 0) &
        np.isfinite(apmag) &
        (fmasked >= 0) & (fmasked < fmasked_thresh) &
        (ndof > 0)
    )
    print(f'  fig_reliability [{b}]: {good.sum():,}/{len(sub):,} pass '
          f'FMASKED_AP04<{fmasked_thresh} & COG_NDOF>0')

    dm     = cogmag[good] - apmag[good]
    rchi2  = np.log10(np.clip(chi2[good] / ndof[good], 1e-3, None))
    ndof_g = ndof[good]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    panels = [
        (rchi2,  r'log$_{10}$(COG $\chi^2_\nu$)'),
        (ndof_g, r'COG$_{\rm NDOF}$'),
    ]
    for ax, (x, xlabel) in zip(axes, panels):
        hb = ax.hexbin(x, dm, gridsize=80, bins='log', cmap='viridis',
                        mincnt=1, linewidths=0)
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(r'COG$_{\rm MTOT}$ $-$ $m_{\rm AP04}$ (mag)')
    plt.colorbar(hb, ax=axes[-1], label='log$_{10}$(N / bin)')

    fig.suptitle(f'SGA-2025  {band.lower()}-band  N={good.sum():,}  '
                 f'(FMASKED$_{{\\rm AP04}}$<{fmasked_thresh})')
    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png or f'phot-reliability-{b.lower()}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_masking(cat, band='R', png=None):
    """FMASKED_AP03/AP04 distributions and their effect on COG-AP agreement."""
    _, colors = plot_style()
    b = band.upper()
    sub = get_sample(cat, band=b)

    fmasked03 = np.asarray(sub[f'FMASKED_AP03_{b}'], dtype=float)
    fmasked04 = np.asarray(sub[f'FMASKED_AP04_{b}'], dtype=float)
    cogmag    = np.asarray(sub[f'COG_MTOT_{b}'], dtype=float)
    apmag     = flux2mag(sub[f'FLUX_AP04_{b}'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bins = np.linspace(0, 1, 51)
    for fm, label, color in [(fmasked03, 'AP03 (1.5x)', colors[0]),
                              (fmasked04, 'AP04 (2.0x)', colors[1])]:
        good = np.isfinite(fm) & (fm >= 0)
        axes[0].hist(fm[good], bins=bins, histtype='step', lw=1.5,
                     color=color, label=f'{label}  N={good.sum():,}')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Masked pixel fraction')
    axes[0].set_ylabel('N')
    axes[0].legend(fontsize=10)

    for thresh in [0.1, 0.2, 0.5]:
        gg = np.isfinite(fmasked04) & (fmasked04 >= 0)
        frac = (fmasked04[gg] < thresh).mean()
        print(f'  FMASKED_AP04_{b} < {thresh}: {frac:.1%} of sample')

    good = (
        np.isfinite(cogmag) & (cogmag > 0) &
        np.isfinite(apmag) &
        (fmasked04 >= 0)
    )
    dm = cogmag[good] - apmag[good]
    hb = axes[1].hexbin(fmasked04[good], dm, gridsize=80, bins='log',
                        cmap='viridis', mincnt=1, linewidths=0)
    axes[1].axhline(0, color='k', ls='--', lw=1)
    axes[1].set_xlabel(r'FMASKED$_{\rm AP04}$')
    axes[1].set_ylabel(r'COG$_{\rm MTOT}$ $-$ $m_{\rm AP04}$ (mag)')
    plt.colorbar(hb, ax=axes[1], label='log$_{10}$(N / bin)')

    fig.suptitle(f'SGA-2025  {band.lower()}-band  N={good.sum():,}')
    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png or f'phot-masking-{b.lower()}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


APERTURES = ['00', '01', '02', '03', '04']
APERTURE_MULT = {'00': 0.5, '01': 1.0, '02': 1.25, '03': 1.5, '04': 2.0}
APERTURES_FOCUS = ['01', '03', '04']


def fig_aperture_series(cat, band='R', apertures=APERTURES_FOCUS,
                        maglim=(9, 21), size_col='D26', size_bins=25, png=None):
    """Row 1: m_APxx vs. COG_MTOT (Hess contours + outliers).
    Row 2: COG_MTOT - m_APxx running median/IQR vs. size_col.
    """
    import seaborn as sns
    import matplotlib.patheffects as pe

    _, colors = plot_style()
    b = band.upper()
    sub = get_sample(cat, band=b)

    cogmag   = np.asarray(sub[f'COG_MTOT_{b}'], dtype=float)
    size     = np.asarray(sub[size_col], dtype=float)
    sizeunit = 'arcsec' if size_col == 'SMA_MOMENT' else 'arcmin'

    fig, axes = plt.subplots(2, len(apertures), figsize=(5.5 * len(apertures), 9.5),
                             sharex='row', sharey='row')
    for col, (ap, color) in enumerate(zip(apertures, colors)):
        apmag = flux2mag(sub[f'FLUX_AP{ap}_{b}'])

        good = np.isfinite(cogmag) & (cogmag > 0) & np.isfinite(apmag)
        print(f'  fig_aperture_series [{b} AP{ap}]: {good.sum():,}/{len(sub):,}')

        x, y = cogmag[good], apmag[good]

        ax0 = axes[0, col]
        hess_contours(ax0, x, y, xrange=maglim, yrange=maglim,
                      cmap=sns.light_palette(color, as_cmap=True))
        ax0.plot(maglim, maglim, color='k', ls='--', lw=1.5, zorder=4,
                 path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        ax0.set_xlim(maglim)
        ax0.set_ylim(maglim)
        ax0.set_title(f'AP{ap} ({APERTURE_MULT[ap]}' + r'$\times$SMA$_{\rm mom}$)')
        ax0.set_xlabel(r'COG$_{\rm MTOT}$ (mag)')

        ax1 = axes[1, col]
        good2 = np.isfinite(cogmag) & (cogmag > 0) & np.isfinite(apmag) & (size > 0)
        dm      = cogmag[good2] - apmag[good2]
        logsize = np.log10(size[good2])

        centers, med, lo, hi = running_binstat(logsize, dm, bins=size_bins)
        ax1.fill_between(centers, lo, hi, color=color, alpha=0.3, linewidth=0)
        ax1.plot(centers, med, color=color, lw=2)
        ax1.axhline(0, color='k', ls='--', lw=1)
        ax1.set_xlabel(rf'log$_{{10}}$ {size_col} ({sizeunit})')

    axes[0, 0].set_ylabel(r'$m_{\rm AP}$ (mag)')
    axes[1, 0].set_ylabel(r'COG$_{\rm MTOT}$ $-$ $m_{\rm AP}$ (mag)')

    fig.suptitle(f'{band.lower()}-band, M=1, SAMPLE=0, 0.5<D26<5 arcmin '
                 f'(N={len(sub):,})')
    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png or f'phot-aperture-series-{b.lower()}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_moment_vs_d26(cat, band='R', bins=25, png='phot-moment-vs-d26.png'):
    """SMA_MOMENT / R26 vs. D26: does the moment-based size scale sub-linearly
    with the isophotal size, explaining the aperture-residual trend with D26?
    """
    _, colors = plot_style()
    sub = get_sample(cat, band=band)

    d26 = np.asarray(sub['D26'], dtype=float)          # arcmin (diameter)
    sma = np.asarray(sub['SMA_MOMENT'], dtype=float)    # arcsec (semi-major axis)
    r26 = d26 * 30.                                     # (D26/2) x 60 -> R26 in arcsec

    good = np.isfinite(sma) & (sma > 0) & np.isfinite(d26) & (d26 > 0)
    logd26 = np.log10(d26[good])
    ratio  = sma[good] / r26[good]

    fig, ax = plt.subplots(figsize=(7, 6))
    yrange = tuple(np.nanpercentile(ratio, [0.5, 99.5]))
    hess_contours(ax, logd26, ratio, xrange=(np.log10(0.5), np.log10(5)), yrange=yrange)

    centers, med, lo, hi = running_binstat(logd26, ratio, bins=bins)
    ax.fill_between(centers, lo, hi, color=colors[0], alpha=0.3, linewidth=0)
    ax.plot(centers, med, color=colors[0], lw=2)

    ax.set_xlabel(r'log$_{10}$ D26 (arcmin)')
    ax.set_ylabel(r'SMA$_{\rm mom}$ / R$_{26}$')
    ax.set_ylim(0.3, 1.3)

    fig.suptitle(f'{band.lower()}-band, M=1, SAMPLE=0, 0.5<D26<5 arcmin '
                 f'(N={len(sub):,})')
    fig.tight_layout()
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_colors(cat, fmasked_thresh=0.2, png='phot-colors.png'):
    """COG-based vs. AP04-based color, for a handful of science-relevant colors."""
    plot_style()
    sub = get_sample(cat, band='R')

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (b1, b2) in zip(axes.flat, COLOR_PAIRS):
        cog1 = np.asarray(sub[f'COG_MTOT_{b1}'], dtype=float)
        cog2 = np.asarray(sub[f'COG_MTOT_{b2}'], dtype=float)
        ap1  = flux2mag(sub[f'FLUX_AP04_{b1}'])
        ap2  = flux2mag(sub[f'FLUX_AP04_{b2}'])
        fm1  = np.asarray(sub[f'FMASKED_AP04_{b1}'], dtype=float)
        fm2  = np.asarray(sub[f'FMASKED_AP04_{b2}'], dtype=float)

        cogcolor = cog1 - cog2
        apcolor  = ap1 - ap2

        good = (
            np.isfinite(cogcolor) & (cog1 > 0) & (cog2 > 0) &
            np.isfinite(apcolor) &
            (fm1 >= 0) & (fm1 < fmasked_thresh) &
            (fm2 >= 0) & (fm2 < fmasked_thresh)
        )
        print(f'  fig_colors [{b1}-{b2}]: {good.sum():,}/{len(sub):,} pass')

        hb = ax.hexbin(apcolor[good], cogcolor[good], gridsize=80, bins='log',
                        cmap='viridis', mincnt=1, linewidths=0)
        if good.sum() > 10:
            lo, hi = np.nanpercentile(apcolor[good], [1, 99])
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel(f'({b1}$-${b2})$_{{\\rm AP04}}$')
        ax.set_ylabel(f'({b1}$-${b2})$_{{\\rm COG}}$')

    fig.suptitle(f'SGA-2025  N$_{{\\rm base}}$={len(sub):,}  '
                 f'(FMASKED$_{{\\rm AP04}}$<{fmasked_thresh} in both bands)')
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
        description='Exploratory SGA-2025 integrated-photometry diagnostics.')
    parser.add_argument('--reliability',     action='store_true')
    parser.add_argument('--masking',         action='store_true')
    parser.add_argument('--aperture-series', action='store_true')
    parser.add_argument('--moment-vs-d26',   action='store_true')
    parser.add_argument('--colors',          action='store_true')
    parser.add_argument('--all',             action='store_true', help='Run all figures')
    parser.add_argument('--band', default='R', help='Optical band for single-band figures')
    parser.add_argument('--fmasked-thresh', type=float, default=0.2,
                        help='FMASKED_AP04 cut used as a reliability threshold')
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    run_all = args.all or not any([
        args.reliability, args.masking, args.aperture_series,
        args.moment_vs_d26, args.colors,
    ])

    cat = read_catalogs()
    print(f'Loaded {len(cat):,} galaxies total')

    if args.reliability or run_all:
        fig_reliability(cat, band=args.band, fmasked_thresh=args.fmasked_thresh)

    if args.masking or run_all:
        fig_masking(cat, band=args.band)

    if args.aperture_series or run_all:
        fig_aperture_series(cat, band=args.band)

    if args.moment_vs_d26 or run_all:
        fig_moment_vs_d26(cat, band=args.band)

    if args.colors or run_all:
        fig_colors(cat, fmasked_thresh=args.fmasked_thresh)


if __name__ == '__main__':
    main()
