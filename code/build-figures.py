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
import numpy.ma as ma
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
# Sky-map helpers (ported from SGA-2020 figures script; no desiutil dependency)
# ---------------------------------------------------------------------------

def _prepare_data(data):
    """Convert to masked array and attach vmin/vmax for colormap normalization."""
    if not isinstance(data, ma.MaskedArray):
        arr  = np.asarray(data, dtype=float)
        data = ma.MaskedArray(arr, mask=(~np.isfinite(arr)) | (arr == 0))
    good     = np.asarray(data[~data.mask])
    data.vmin = float(good.min()) if len(good) else 0.
    data.vmax = float(good.max()) if len(good) else 1.
    return data


def _init_sky(ra_center=120., galactic_plane_color='k', ax=None):
    """Set up a Mollweide axes with projection helpers and galactic plane."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if ax is None:
        ax = plt.axes(projection='mollweide')

    ax._ra_center = ra_center

    def projection_ra(ra):
        r = np.remainder(np.atleast_1d(np.asarray(ra, float)) + 360 - ra_center, 360)
        r[r > 180] -= 360
        return np.radians(-r)

    def projection_dec(dec):
        return np.radians(np.atleast_1d(np.asarray(dec, float)))

    ax.projection_ra  = projection_ra
    ax.projection_dec = projection_dec

    if galactic_plane_color and galactic_plane_color != 'none':
        l_gp = np.linspace(0, 360, 1000)

        def _draw_gal_line(b_val, lw):
            gc = SkyCoord(l=l_gp * u.deg, b=np.full(1000, b_val) * u.deg, frame='galactic')
            x  = projection_ra(gc.icrs.ra.degree)
            y  = projection_dec(gc.icrs.dec.degree)
            for seg in np.split(np.arange(len(x)), np.where(np.abs(np.diff(x)) > np.pi)[0] + 1):
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], '-', color=galactic_plane_color, lw=lw, zorder=2)

        _draw_gal_line(0.,   lw=1.0)   # Galactic plane (b = 0)
        _draw_gal_line(+10., lw=0.5)   # |b| = 10 exclusion boundary
        _draw_gal_line(-10., lw=0.5)

    ax.grid(True, color='0.6', linewidth=0.4, linestyle='-', alpha=0.7, zorder=1)

    # Replace the default −180…+180 longitude labels with 0…360 RA labels.
    # x_rad is the raw Mollweide x coordinate in radians; inverting projection_ra gives:
    #   RA = (ra_center − degrees(x_rad)) % 360
    from matplotlib.ticker import FuncFormatter
    def _ra_label(x_rad, pos):
        ra = int(round((ra_center - np.degrees(x_rad)) % 360))
        return f'{ra % 360}°'
    ax.xaxis.set_major_formatter(FuncFormatter(_ra_label))

    return ax


def _ar_sky_cbar(ax, sc, label, extend=None, mloc=None):
    """Horizontal colorbar below a sky map (adapted from SGA-2020)."""
    cbar = plt.colorbar(sc, ax=ax, location='bottom', orientation='horizontal',
                        spacing='proportional', extend=extend, extendfrac=0.025,
                        pad=0.1, fraction=0.035, aspect=40)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(label)
    if mloc is not None:
        cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(mloc))


def _plot_healpix_map(data, nest=False, cmap='viridis', colorbar=True,
                      label=None, ax=None):
    """Plot a healpix density map on an existing Mollweide axes (adapted from SGA-2020)."""
    import healpy as hp
    from matplotlib.colors import Normalize
    from matplotlib.collections import PolyCollection

    data  = _prepare_data(data)
    nside = hp.npix2nside(len(data))

    if ax is None:
        ax = _init_sky()

    proj_edge = ax._ra_center - 180
    while proj_edge < 0:
        proj_edge += 360

    corners               = hp.boundaries(nside, np.arange(len(data)), step=1, nest=nest)
    corner_theta, corner_phi = hp.vec2ang(corners.transpose(0, 2, 1))
    corner_ra  = np.degrees(corner_phi)
    corner_dec = np.degrees(np.pi / 2 - corner_theta)

    x = ax.projection_ra(corner_ra)
    y = ax.projection_dec(corner_dec)

    verts = np.array([x.reshape(-1, 4), y.reshape(-1, 4)]).transpose(1, 2, 0)

    uv_verts   = np.array([corner_phi.reshape(-1, 4),
                           corner_theta.reshape(-1, 4)]).transpose(1, 2, 0)
    theta_edge = np.unique(uv_verts[:, :, 1])
    phi_edge   = np.radians(proj_edge)
    eps        = 0.1 * np.sqrt(hp.nside2pixarea(nside))
    wrapped    = np.unique(np.hstack([
        hp.ang2pix(nside, theta_edge, phi_edge - eps, nest=nest),
        hp.ang2pix(nside, theta_edge, phi_edge + eps, nest=nest),
    ]))
    data.mask[wrapped] = True

    try:
        norm = Normalize(vmin=data.vmin, vmax=data.vmax)
    except AttributeError:
        norm = None

    good       = np.where(~data.mask)[0]
    collection = PolyCollection(verts[good], array=data[good], cmap=cmap,
                                norm=norm, edgecolors='none', zorder=0)
    ax.add_collection(collection)
    ax.autoscale_view()

    if colorbar:
        bar = plt.colorbar(collection, ax=ax, orientation='horizontal',
                           spacing='proportional', pad=0.11, fraction=0.05, aspect=50)
        if label:
            bar.set_label(label)

    return ax


def _plot_sky_binned(ra, dec, max_bin_area=10, clip_lo=0, verbose=False,
                     cmap='viridis', colorbar=False, ax=None, return_grid_data=False):
    """Bin galaxies into HEALPix pixels and plot surface density (adapted from SGA-2020)."""
    import healpy as hp

    ra  = np.asarray(ra,  dtype=float).ravel()
    dec = np.asarray(dec, dtype=float).ravel()

    for n in range(1, 25):
        nside    = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break
    npix = hp.nside2npix(nside)
    nest = False
    if verbose:
        print(f'  HEALPix NSIDE={nside}, pixel area={bin_area:.3f} deg²')

    pixels    = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest)
    counts    = np.bincount(pixels, minlength=npix).astype(float)
    grid_data = _prepare_data(counts / bin_area)

    # mask empty pixels and any below clip_lo
    grid_data[ma.getmaskarray(grid_data) | (grid_data <= clip_lo)] = ma.masked

    ax = _plot_healpix_map(grid_data, nest=nest, cmap=cmap,
                           colorbar=colorbar, ax=ax)
    if return_grid_data:
        return ax, grid_data
    return ax


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def plot_style(font_scale=0.8, paper=False, talk=True):
    """Seaborn plot style wrapper (ported from SGA-2020 figures script)."""
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


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_sky(cat, title='SGA-2025', png='sga2025-sky.png',
            ra_center=120., max_bin_area=10, clip_lo=0, cmap='twilight'):
    """Mollweide sky map of galaxy surface density with galactic plane overlay."""
    plot_style()

    n_gal = len(np.unique(np.asarray(cat['SGAID'])))

    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax  = fig.add_subplot(111, projection='mollweide')
    ax  = _init_sky(ra_center=ra_center, galactic_plane_color='k', ax=ax)

    ax, grid_data = _plot_sky_binned(
        cat['RA'].data.astype(float), cat['DEC'].data.astype(float),
        max_bin_area=max_bin_area, clip_lo=clip_lo, verbose=True,
        cmap=cmap, colorbar=False, ax=ax, return_grid_data=True,
    )
    print(f'  Surface density: median={float(ma.median(grid_data)):.1f}  '
          f'mean={float(ma.mean(grid_data)):.1f}  '
          f'std={float(ma.std(grid_data)):.1f}  gal/deg²')

    # Grid and RA labels drawn last, above the healpix polygons
    ax.grid(True, color='0.6', linewidth=0.4, linestyle='-', alpha=0.7, zorder=3)
    ax.xaxis.set_zorder(10)

    ax.set_title(f'{title} (N={n_gal:,})', pad=20)#, fontsize=15, fontweight='bold')
    ax.set_xlabel('RA (degrees)', labelpad=10)
    ax.set_ylabel('Dec (degrees)')

    _ar_sky_cbar(ax, ax.collections[-1], r'Galaxy Surface Density (deg$^{-2}$)',
                 extend='both', mloc=10)

    fig.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.93)
    outfile = os.path.join(FIG_DIR, png)
    fig.savefig(outfile)
    print(f'Wrote {outfile}')
    plt.close(fig)


def fig_sample(cat, png='sga2025-sample.png'):
    """D26 diameter distribution for both imaging regions."""
    _, colors = plot_style()

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
    _, colors = plot_style()

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
    _, colors = plot_style()

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
    _, colors = plot_style()

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
    _, colors = plot_style()

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
        south = cat[cat['REGION_LABEL'] == 'dr11-south']
        north = cat[cat['REGION_LABEL'] == 'dr11-north']
        # deduplicate combined by SGAID (any object appearing in both regions is counted once)
        _, idx = np.unique(np.asarray(cat['SGAID']), return_index=True)
        combined = cat[np.sort(idx)]
        fig_sky(south,    title='SGA-2025 DR11 South', png='sga2025-sky-south.png')
        fig_sky(north,    title='SGA-2025 DR11 North', png='sga2025-sky-north.png')
        fig_sky(combined, title='SGA-2025',            png='sga2025-sky.png')

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
