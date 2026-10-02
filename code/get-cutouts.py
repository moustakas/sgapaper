#!/usr/bin/env python
"""Select candidate galaxies for the overview figure and download their
public thumbnails.

Usage examples
--------------
python code/get-cutouts.py --dry-run
python code/get-cutouts.py
python code/get-cutouts.py --ncand 5 --seed 2

Thumbnails and a manifest (cutouts.csv) are written to data/cutouts/, which
is git-ignored. Catalogs are read via SGA.SGA.read_sga_sample, which
requires $SGA_PUBLIC_DIR to be set.
"""
import os
import time
import argparse
import urllib.request
import urllib.error

import numpy as np
from astropy.table import Table, vstack, unique

from SGA.SGA import read_sga_sample
from SGA.coadds import REGIONBITS

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOUT_DIR = os.path.join(REPO_DIR, 'data', 'cutouts')

HTML_URL = 'https://portal.nersc.gov/project/cosmo/sga/2025/html'

# Target angular diameters (arcsec) and absolute magnitudes; keep in sync
# with fig_overview in build-figures.py.
THETAS  = (600., 180., 60., 30., 15., 8.)
ABSMAGS = (-21., -18.)


def thumbnail_url(group_name, region):
    """URL of the public thumbnail for one group.

    Parameters
    ----------
    group_name : :class:`str`
        Group name, e.g., ``00000p3173``.
    region : :class:`str`
        Imaging region, ``dr11-south`` or ``dr11-north``.

    Returns
    -------
    :class:`str`
        Thumbnail URL.

    """
    return (f'{HTML_URL}/{region}/{group_name[:3]}/{group_name}/'
            f'SGA2025_{group_name}-thumb.jpg')


def dered_mag(cat, band='R'):
    """Extinction-corrected AB magnitude from the nominal flux.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        SGA-2025 catalog.
    band : :class:`str`
        Bandpass, e.g., ``R``.

    Returns
    -------
    :class:`numpy.ndarray`
        Magnitudes; NaN where the flux is not positive.

    """
    flux = np.asarray(cat[f'FLUX_{band}'], dtype=float)
    mwtrans = np.asarray(cat[f'MW_TRANSMISSION_{band}'], dtype=float)
    mag = np.full(len(cat), np.nan)
    good = (flux > 0) & (mwtrans > 0)
    mag[good] = 22.5 - 2.5 * np.log10(flux[good] / mwtrans[good])
    return mag


def select_candidates(cat, thetas=THETAS, absmags=ABSMAGS, dtheta=0.15,
                      dabsmag=0.3, ncand=3, seed=1):
    """Select candidate galaxies in bins of angular diameter and luminosity.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Full SGA-2025 catalog.
    thetas : :class:`tuple`
        Target angular diameters (arcsec).
    absmags : :class:`tuple`
        Target r-band absolute magnitudes.
    dtheta : :class:`float`
        Fractional half-width of each angular-diameter bin.
    dabsmag : :class:`float`
        Half-width of each absolute-magnitude bin (mag).
    ncand : :class:`int`
        Number of candidates to draw per bin.
    seed : :class:`int`
        Random seed for drawing candidates.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per candidate.

    """
    rng = np.random.default_rng(seed)

    dist = np.asarray(cat['DIST'], dtype=float)
    rmag = dered_mag(cat, 'R')
    gmag = dered_mag(cat, 'G')
    d26  = np.asarray(cat['D26'], dtype=float) * 60. # [arcsec]

    clean = ((np.asarray(cat['SAMPLE']) == 0) &
             (np.asarray(cat['ELLIPSEBIT']) == 0) &
             (np.asarray(cat['ELLIPSEMODE']) == 0) &
             (np.asarray(cat['GROUP_MULT']) == 1) &
             (np.asarray(cat['DIST_IVAR']) > 0) & (dist > 0) &
             np.isfinite(rmag) & (d26 > 0))
    print(f'Clean, isolated sample with a distance: {clean.sum():,} / {len(cat):,}')
    print(f'  D26 range: {d26[clean].min():.1f}-{d26[clean].max():.1f} arcsec')

    with np.errstate(divide='ignore', invalid='ignore'):
        absmag = rmag - 5. * np.log10(dist) - 25.
    gr = gmag - rmag

    rows = []
    for absmag0 in absmags:
        for theta in thetas:
            inbin = np.where(clean & (np.abs(absmag - absmag0) < dabsmag) &
                             (np.abs(d26 / theta - 1.) < dtheta))[0]
            print(f'  M_r={absmag0:.1f}, D26={theta:g} arcsec: {len(inbin):,} galaxies')
            if len(inbin) == 0:
                continue
            for indx in rng.choice(inbin, size=min(ncand, len(inbin)), replace=False):
                one = cat[indx]
                region = 'dr11-south' if (one['REGION'] & REGIONBITS['dr11-south']) != 0 else 'dr11-north'
                group_name = str(one['GROUP_NAME']).strip()
                rows.append(dict(
                    ABSMAG_TARGET=absmag0, THETA_TARGET=theta,
                    SGAID=one['SGAID'], SGANAME=str(one['SGANAME']).strip(),
                    GALAXY=str(one['GALAXY']).strip(), GROUP_NAME=group_name,
                    REGION=region, RA=one['RA'], DEC=one['DEC'],
                    D26=np.round(d26[indx], 2), BA=one['BA'], PA=one['PA'],
                    Z=one['Z'], DIST=np.round(dist[indx], 3),
                    ABSMAG_R=np.round(absmag[indx], 3), GR=np.round(gr[indx], 3),
                    URL=thumbnail_url(group_name, region),
                    FILE=f'SGA2025_{group_name}-thumb.jpg'))

    return Table(rows=rows)


def download(cand, maxquery=50, sleep=0.5, clobber=False):
    """Download thumbnails, skipping files already on disk.

    Parameters
    ----------
    cand : :class:`astropy.table.Table`
        Candidates from :func:`select_candidates`.
    maxquery : :class:`int`
        Maximum number of requests to make.
    sleep : :class:`float`
        Pause between requests (s).
    clobber : :class:`bool`
        Re-download files already on disk.

    """
    nquery = 0
    for one in cand:
        outfile = os.path.join(CUTOUT_DIR, one['FILE'])
        if os.path.isfile(outfile) and not clobber:
            continue
        if nquery >= maxquery:
            print(f'Reached the limit of {maxquery} requests; stopping.')
            break
        nquery += 1
        try:
            urllib.request.urlretrieve(one['URL'], outfile)
            print(f'Wrote {outfile}')
        except (urllib.error.HTTPError, urllib.error.URLError) as err:
            print(f'Failed: {one["URL"]} ({err})')
        time.sleep(sleep)
    print(f'Made {nquery} requests.')


def main():
    parser = argparse.ArgumentParser(
        description='Download candidate cutouts for the overview figure.')
    parser.add_argument('--ncand', type=int, default=3,
                        help='Candidates per (M_r, diameter) bin')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--maxquery', type=int, default=50,
                        help='Maximum number of requests')
    parser.add_argument('--dry-run', action='store_true',
                        help='Select candidates and write the manifest only')
    parser.add_argument('--clobber', action='store_true')
    args = parser.parse_args()

    os.makedirs(CUTOUT_DIR, exist_ok=True)

    _, cat = read_sga_sample(beta=False, verbose=True)
    cand = select_candidates(cat, ncand=args.ncand, seed=args.seed)
    if len(cand) == 0:
        print('No candidates selected.')
        return

    # accumulate candidates across seeds unless clobbering
    manifest = os.path.join(CUTOUT_DIR, 'cutouts.csv')
    if os.path.isfile(manifest) and not args.clobber:
        cand = unique(vstack([Table.read(manifest, format='ascii.csv'), cand]),
                      keys='SGAID', keep='first')
    cand.sort(['ABSMAG_TARGET', 'THETA_TARGET', 'SGAID'])
    cand.write(manifest, format='ascii.csv', overwrite=True)
    print(f'Wrote {manifest} ({len(cand)} candidates)')

    if not args.dry_run:
        download(cand, maxquery=args.maxquery, clobber=args.clobber)


if __name__ == '__main__':
    main()
