from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from ase.build import mx2
from ase import Atoms
from sklearn.neighbors import NearestNeighbors


def crop_atoms(atoms, L):
    pts = atoms.positions
    x, y, z = pts.T
    l = L
    mask1 = np.logical_and(x > 0, x < l)
    mask2 = np.logical_and(y > 0, y < l)
    mask = mask1 * mask2
    pts_ = pts[mask]
    numbers_ = atoms.numbers[mask]
    return Atoms(numbers=numbers_, positions=pts_, cell=atoms.cell)

class VacRandomModel:
    """
    A model to generate and analyze random vacancies and dopants in a 2D material.

    Attributes:
        a (float): Lattice constant.
        area (float): Total area of the 2D material.
        num_vac (int): Number of vacancies.
        vac (np.ndarray): Coordinates of the vacancies.
        x (np.ndarray): Coordinates of the remaining atoms after removing vacancies.
        num_dopant (int): Number of dopants.
        dopant (np.ndarray): Coordinates of the dopants.
        m (np.ndarray): Coordinates of the remaining atoms after removing dopants.
    """

    def __init__(self, a: float = 3.57, size: int = 100, density: float = 1e13, dopant_scale: float = 2.12, crop = False) -> None:
        """
        Initializes the VacRandomModel with the given parameters.

        Args:
            a (float): Lattice constant. Default is 3.57.
            size (int): Size of the 2D material. Default is 100.
            density (float): Vacancy density. Default is 1e13.
            dopant_scale (float): Scale factor for dopants. Default is 2.12.
        """
        self.a = a
        self.area = 0.5 * np.sqrt(3) * (size * a) ** 2
        self.num_vac = np.round(self.area * density / (1e8) / (1e8)).astype(int)

        # ase mx2 to build
        atoms = mx2(a=a, size=(size, size, 1))
        if crop:
            atoms = crop_atoms(atoms, L = a * size * 0.5)
            self.area = (0.5 * size * a) ** 2
        else:
            self.area = 0.5 * np.sqrt(3) * (size * a) ** 2
        self.num_vac = np.round(self.area * density / (1e8) / (1e8)).astype(int)

        pts = atoms.get_positions()
        mask1 = pts[:, 2] > 0
        mask2 = pts[:, 2] == 0
        atoms1 = atoms[mask1]
        atoms2 = atoms[mask2]
        # pts1 are x2 sites, pts2 are m sites
        pts1 = atoms1.get_positions()
        pts2 = atoms2.get_positions()

        # Random select to generate vac
        ind = np.random.choice(pts1.shape[0], self.num_vac, replace=False)
        mask = np.zeros(len(pts1)).astype(bool)
        mask[ind] = True
        self.vac = pts1[mask]
        self.x = pts1[~mask]

        # Random select to generate dopant
        self.num_dopant = np.ceil(self.num_vac / dopant_scale).astype(int)
        ind = np.random.choice(pts2.shape[0], self.num_dopant, replace=False)
        mask = np.zeros(len(pts2)).astype(bool)
        mask[ind] = True
        self.dopant = pts2[mask]
        self.m = pts2[~mask]

    def get_knn_d(self) -> np.ndarray:
        """
        Computes the nearest neighbor distances among vacancies.

        Returns:
            np.ndarray: Counts of unique distances between vacancies, focusing on the first three counts.
        """
        pts = self.vac
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pts)
        d, ind = nbrs.kneighbors(pts, return_distance=True)
        d = d[:, 1]
        d = np.round(d, 4)
        dist, cnts = np.unique(d, return_counts=True)
        return cnts[0:3]

    def count_dopant_vac_pairs(self) -> Tuple[int, int]:
        """
        Counts the number of dopant-vacancy pairs within a specified radius.

        Returns:
            Tuple[int, int]: Total number of dopant-vacancy pairs and the total number of dopants.
        """
        nbrs = NearestNeighbors(radius=1.5 * self.a, algorithm='ball_tree').fit(self.dopant)
        d, ind = nbrs.radius_neighbors(self.vac)
        pair_counts = np.sum([len(e) for e in d])
        return pair_counts, len(self.dopant)

    def plot(self, ax: plt.Axes = None, **kwargs) -> None:
        """
        Plots the positions of vacancies, dopants, and remaining atoms.

        Args:
            ax (plt.Axes, optional): Matplotlib axes object. Default is None.
            **kwargs: Additional keyword arguments for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        s = 3
        ax.scatter(self.m[:, 0], self.m[:, 1], color='r', s=s)
        ax.scatter(self.x[:, 0], self.x[:, 1], color='b', s=s)
        ax.scatter(self.vac[:, 0], self.vac[:, 1], color='g', s=50)
        ax.scatter(self.dopant[:, 0], self.dopant[:, 1], color='k', s=50)
        ax.axis('equal')
