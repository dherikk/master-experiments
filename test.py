import numpy as np
import matplotlib.pyplot as plt
from gaussian import gaussian
from typing import Union, Any
from hexagon_grid import hexagon_grid
from filter_bank import filter_bank
import multiprocessing

def experiment6(X:np.ndarray[Union[Any, np.floating]], sig:float=20., length: int=6, rings: int=3, threshhold:float = 0.5) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(length*hexagon_grid(rings))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_5, [(X, filters_real, filters_abs, theta_indices, p, threshhold) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

    def calc_5(data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, theta_indices: list[list[int]], center: np.ndarray, threshhold: float) -> np.ndarray:
    coefficients = -np.ones((data.shape[0], filters_real.shape[0]))
    for dy in np.arange(-3, 4):
        for dx in np.arange(-3, 4):
            rolled = np.roll(data, np.rint(center + [-dx, dy]).astype(int), axis=(2, 1))
            for i in range(filters_real.shape[0]):
                masked = rolled * filters_abs[i]
                masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))

    final_coefs =  np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    
    return np.where(final_coefs > threshhold, final_coefs, 0.)