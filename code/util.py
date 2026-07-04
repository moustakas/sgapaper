"""Utility functions.

"""
import numpy as np

def hess_contours(ax, x, y, xrange, yrange, bins=50, smooth=1.0,
                  contour_levels=None, cmap='Blues', contour_lw=1.5,
                  contour_color='k', outlier_ms=2, background=True):
    """Hess diagram with smoothed cumulative contours, matching corner_plot style.

    Points that fall outside the outermost contour are drawn as individual small
    markers (size ``outlier_ms``); pass ``outlier_ms=0`` to suppress them.
    The pcolormesh background is rasterized so PDF output stays compact.
    Pass ``background=False`` to draw contours and outlier scatter only (no
    pcolormesh), which is useful when overlaying multiple samples on one axes.
    """
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    if contour_levels is None:
        contour_levels = [0.5, 0.75, 0.95, 0.995]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                        range=[xrange, yrange])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    if background:
        ax.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1), cmap=cmap,
                      rasterized=True, zorder=1)

    Hs = gaussian_filter(H, smooth) if smooth > 0 else H
    flat = np.sort(Hs.flatten())[::-1]
    cumsum = np.cumsum(flat)
    total = cumsum[-1]
    lvls = []
    if total > 0 and contour_levels:
        for frac in contour_levels:
            idx = np.searchsorted(cumsum, frac * total)
            lvls.append(flat[min(idx, len(flat) - 1)])
        lvls = sorted(v for v in set(lvls) if v > 0)
        if lvls:
            ax.contour(xc, yc, Hs.T, levels=lvls,
                       colors=contour_color, linewidths=contour_lw, zorder=3)

    # individual points outside the outermost contour, drawn above the Hess
    # background (zorder=2) but below contours (zorder=3)
    if lvls and outlier_ms > 0:
        in_range = ((x >= xrange[0]) & (x <= xrange[1]) &
                    (y >= yrange[0]) & (y <= yrange[1]))
        xi = np.clip(np.digitize(x[in_range], xedges) - 1, 0, len(xc) - 1)
        yi = np.clip(np.digitize(y[in_range], yedges) - 1, 0, len(yc) - 1)
        outside = Hs[xi, yi] < lvls[0]
        ax.scatter(x[in_range][outside], y[in_range][outside],
                   s=outlier_ms, c=[contour_color], alpha=0.8,
                   linewidths=0, zorder=2, rasterized=True)


def running_binstat(x, y, bins=20, xrange=None, q_lo=25, q_hi=75, min_count=5):
    """Bin y on x; return bin centers, running median, and percentile bounds.

    Parameters
    ----------
    x, y : array_like
        Input arrays (must be 1-D and the same length; should be finite).
    bins : int
        Number of equal-width bins.
    xrange : (float, float) or None
        Range of x. Passed to binned_statistic; defaults to (x.min(), x.max()).
    q_lo, q_hi : float
        Lower and upper percentiles for the shaded band (default: 25, 75 → IQR).
    min_count : int
        Bins with fewer than this many points are set to NaN.

    Returns
    -------
    centers : ndarray, shape (bins,)
        Bin midpoints.
    median : ndarray, shape (bins,)
        Median of y in each bin.
    p_lo, p_hi : ndarray, shape (bins,)
        q_lo-th and q_hi-th percentiles of y in each bin.
        Bins with count < min_count contain NaN.
    """
    from scipy.stats import binned_statistic

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    kw = dict(bins=bins, range=xrange)

    cnt, edges, _ = binned_statistic(x, y, statistic='count', **kw)
    med, _, _     = binned_statistic(x, y, statistic='median', **kw)
    lo,  _, _     = binned_statistic(x, y,
                                     statistic=lambda v: np.percentile(v, q_lo), **kw)
    hi,  _, _     = binned_statistic(x, y,
                                     statistic=lambda v: np.percentile(v, q_hi), **kw)

    centers = 0.5 * (edges[:-1] + edges[1:])
    sparse = cnt < min_count
    for arr in (med, lo, hi):
        arr[sparse] = np.nan

    return centers, med, lo, hi
