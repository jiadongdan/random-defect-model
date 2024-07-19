import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_pts_selected(pts, rmax=30):
    xmin = pts[:, 0].min() + rmax
    xmax = pts[:, 0].max() - rmax

    ymin = pts[:, 1].min() + rmax
    ymax = pts[:, 1].max() - rmax

    mask1 = np.logical_and(pts[:, 0] > xmin, pts[:, 0] < xmax)
    mask2 = np.logical_and(pts[:, 1] > ymin, pts[:, 1] < ymax)
    mask = np.logical_and(mask1, mask2)
    area = (xmax - xmin) * (ymax - ymin)
    return pts[mask], area


def get_rdf(pts, rmax=30, bins=150, fixed_bins=False, alpha=0.1):
    pts_, area = get_pts_selected(pts, rmax=rmax)
    nbrs = NearestNeighbors(radius=rmax, algorithm='ball_tree').fit(pts_)
    d, inds = nbrs.radius_neighbors(pts)
    d = np.hstack(d)
    # d = d[d>0]
    N = len(pts_)

    if fixed_bins:
        bins = np.linspace(0, rmax, bins)

    cnts, bin_edges = np.histogram(d, bins=bins)
    R = 0.5 * (bin_edges[1:] + bin_edges[0:-1])
    dr = np.diff(bin_edges)[0]
    rho = area / N
    y = cnts / (2 * np.pi * R * dr) * rho / N
    y[0] = 0
    return R, y

def get_rdf_ij(pts1, pts2, rmax=30, bins=150, fixed_bins=False):
    pts_, area = get_pts_selected(pts1, rmax=rmax)
    nbrs = NearestNeighbors(radius=rmax, algorithm='ball_tree').fit(pts_)
    d, inds = nbrs.radius_neighbors(pts2)
    d = np.hstack(d)
    # d = d[d>0]
    N = len(pts_)

    if fixed_bins:
        bins = np.linspace(0, rmax, bins)

    cnts, bin_edges = np.histogram(d, bins=bins)
    R = 0.5 * (bin_edges[1:] + bin_edges[0:-1])
    dr = np.diff(bin_edges)[0]
    rho = area / N
    y = cnts / (2 * np.pi * R * dr) * rho / N
    y[0] = 0
    return R, y
