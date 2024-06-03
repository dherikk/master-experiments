import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any, Tuple
from components.hexagon_grid import hexagon_grid
from components.filter_bank import filter_bank, filter_bank_img, filter_bank_img_neg
import multiprocessing
from pathos.threading import ThreadPool

def make_t_sne(tsne, y, title):
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout(pad=0.0)
    plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10).tolist())
    plt.title(title, fontsize=12)
    plt.show()

def calc(training_data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, th_i: list[list[int]], theta: Union[np.float64, np.ndarray]) -> np.ndarray:
    coefficients = -np.ones((training_data.shape[0], filters_real.shape[0]))
    for dy in np.arange(-3, 4):
        for dx in np.arange(-3, 4):
            center = [dy, -dx] if np.isnan(theta) else [int(dy + np.rint(7 * np.sin(theta))), -int(dx + np.rint(7 * np.cos(theta)))]
            rolled = np.roll(training_data, np.rint(center).astype(int), axis=(2, 1))
            for i in range(filters_real.shape[0]):
                masked = rolled * filters_abs[i]
                masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))    

    coefficients = np.hstack((coefficients[:, th_i[0]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[2]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[3]].max(axis=1).reshape(-1, 1)))
    return coefficients

## Calc_2 is deprecated in favour of more efficient version calc_4
def calc_2(training_data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, th_i, point) -> np.ndarray:
    coefficients = np.ones((training_data.shape[0], len(filters_real))) * -1
    for image_i, image in enumerate(training_data):
        for filter_i, _ in enumerate(filters_real):
            for dy in np.arange(-3, 4) + point[0]:
                for dx in np.arange(-3, 4) + point[1]:
                    image_masked = np.ones(image.shape)
                    image_masked = np.roll(image, (int(dy), -int(dx)), axis=(0, 1)) * filters_abs[filter_i]
                    image_masked /= np.sqrt(np.sum(np.square(image_masked)))
                    coefficients[image_i, filter_i] = np.max([coefficients[image_i, filter_i], np.sum(image_masked * filters_real[filter_i])])
    coefficients = np.hstack((coefficients[:, th_i[0]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[2]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[3]].max(axis=1).reshape(-1, 1)))
    return coefficients

def calc_3(training_data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, th_i: list[list[int]], cir_i: list[list[int]], point: np.ndarray) -> np.ndarray:
    coefficients = -np.ones((training_data.shape[0], filters_real.shape[0]))

    for dy in np.arange(-3, 4):
        for dx in np.arange(-3, 4):
            rolled = np.roll(training_data, np.rint(point + [-dx, dy]).astype(int), axis=(2, 1))
            for i in range(filters_real.shape[0]):
                masked = rolled * filters_abs[i]
                masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))
    coefficients = np.hstack((coefficients[:, th_i[0]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[2]].max(axis=1).reshape(-1, 1),
                              coefficients[:, th_i[3]].max(axis=1).reshape(-1, 1),
                              coefficients[:, cir_i[0]].max(axis=1).reshape(-1, 1),
                              coefficients[:, cir_i[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, cir_i[2]].max(axis=1).reshape(-1, 1)))
                              
                              
    return coefficients

def calc_4(data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, theta_indices: list[list[int]], center: np.ndarray) -> np.ndarray:
    # New calc which saves memory
    coefficients = -np.ones((data.shape[0], filters_real.shape[0]))

    for dy in np.arange(-3, 4):
        for dx in np.arange(-3, 4):
            rolled = np.roll(data, np.rint(center + [-dx, dy]).astype(int), axis=(2, 1))
            for i in range(filters_real.shape[0]):
                masked = rolled * filters_abs[i]
                masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

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

def calc_6(data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, theta_indices: list[list[int]], center: np.ndarray, grid_size: int) -> np.ndarray:
    coefficients = -np.ones((data.shape[0], filters_real.shape[0]))
    # change arange to reflect grid size
    for dy in np.arange(-grid_size, grid_size + 1):
        for dx in np.arange(-grid_size, grid_size + 1):
            rolled = np.roll(data, np.rint(center + [-dx, dy]).astype(int), axis=(2, 1))
            for i in range(filters_real.shape[0]):
                masked = rolled * filters_abs[i]
                masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

def calc_coefficients(data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, center: np.ndarray, point: np.ndarray) -> np.ndarray:
    filter_dots = -np.ones(filters_real.shape[0])
    rolled = np.roll(data, np.rint(center + point).astype(int), axis=(2, 1))
    for i in range(filters_real.shape[0]):
        masked = rolled * filters_abs[i]
        masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
        filter_dots[i] = np.sum(masked * filters_real[i], axis=(1, 2))

    return filter_dots

def calc_parallel(data: np.ndarray, filters_real: np.ndarray, filters_abs: np.ndarray, theta_indices: list[list[int]], center: np.ndarray) -> np.ndarray:
    # New calc with nested parallelism
    coefficients = -np.ones((data.shape[0], filters_real.shape[0]))
    grid = [np.array([dy, dx]) for dy in np.arange(-3, 4) for dx in np.arange(-3, 4)]
    pool = ThreadPool(len(grid))

    results = pool.imap(lambda p: calc_coefficients(*p), [(data, filters_real, filters_abs, center, point) for point in grid])
    returned = list(results)

    for r in returned:
        for i in range(filters_real.shape[0]):
            coefficients[:, i] = np.maximum(coefficients[:, i], r[i])

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

def create_coefficients(data: np.ndarray, center: np.ndarray, grid_scale: int):
    filters_real, _, filters_abs, theta_indices = filter_bank(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters_real.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters_real.shape[0]):
                    masked = rolled * filters_abs[i]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

def create_coefficients_with_threshold(data: np.ndarray, center: np.ndarray, grid_scale: int, threshhold: float):
    filters_real, _, filters_abs, theta_indices = filter_bank(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters_real.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters_real.shape[0]):
                    masked = rolled * filters_abs[i]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters_real[i], axis=(1, 2)))

    final_coefs = np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    
    return np.where(final_coefs > threshhold, final_coefs, 0.)

def create_coefficients_with_img(data: np.ndarray, center: np.ndarray, grid_scale: int):
    filters_abs, filters, theta_indices = filter_bank_img(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters.shape[0]):
                    masked = rolled * filters_abs[i//3]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters[i], axis=(1, 2)))

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

def create_coefficients_with_img_and_threshold(data: np.ndarray, center: np.ndarray, grid_scale: int, threshold: float):
    filters_abs, filters, theta_indices = filter_bank_img(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters.shape[0]):
                    masked = rolled * filters_abs[i//3]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters[i], axis=(1, 2)))

    final_coefs = np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                             coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    
    return np.where(final_coefs > threshold, final_coefs, 0.)

def create_coefficients_with_img_and_neg(data: np.ndarray, center: np.ndarray, grid_scale: int):
    filters_abs, filters, theta_indices = filter_bank_img_neg(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters.shape[0]):
                    masked = rolled * filters_abs[i//4]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters[i], axis=(1, 2)))

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))

def create_coefficients_with_img_and_threshold_and_neg(data: np.ndarray, center: np.ndarray, grid_scale: int, threshold: float):
    filters_abs, filters, theta_indices = filter_bank_img_neg(27, 20., 2 * 4, 2)

    coefficients = np.full((data.shape[0], filters.shape[0]), -1., dtype=np.float32)

    radius = grid_scale / np.sqrt(3.)  # The radius of the circumscribed circle is 2/sqrt(3) * grid_scale/2
    for d1 in np.arange(-radius, radius + 1., 1.):
        for d2 in np.arange(-radius, radius + 1., 1.):
            d = np.array([d1, d2])
            if np.square(d).sum() <= np.square(radius + 0.15):  # +0.15 to avoid gaps
                rolled = np.roll(data, (center + d).astype(int), axis=(1, 2))
                for i in range(filters.shape[0]):
                    masked = rolled * filters_abs[i//4]
                    masked /= np.sqrt(np.sum(np.square(masked), axis=(1, 2))).reshape(-1, 1, 1)
                    coefficients[:, i] = np.maximum(coefficients[:, i], np.sum(masked * filters[i], axis=(1, 2)))

    final_coefs = np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                             coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    
    return np.where(final_coefs > threshold, final_coefs, 0.)

## 1 Ring
def experiment1(X:np.ndarray, sig:float=20.) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)

    manifold = []
    thetas = np.hstack([[np.NaN], np.arange(0, 2 * np.pi, np.pi / 3)])

    with multiprocessing.Pool() as p:
        manifold += p.starmap(calc, [(X, filters_real, filters_abs, theta_indices, t) for t in thetas])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped
## 2 Rings
def experiment2(X:np.ndarray, sig:float=20.) -> np.ndarray:

    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(8*hexagon_grid(2))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_4, [(X, filters_real, filters_abs, theta_indices, p) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped
## 3 Rings
def experiment4(X:np.ndarray, sig:float=20., grid_scale: int=7, grid_radius: int=3) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients, [(X, p, grid_scale) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped
## 3 Rings + threshold
def experiment6(X:np.ndarray, grid_scale: int=7, grid_radius: int=3, threshold:float = 0.5) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients_with_threshold, [(X, p, grid_scale, threshold) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

""" def experiment7(X:np.ndarray, sig:float=20., length: int=6, rings: int=3, grid_size=3) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(length*hexagon_grid(rings))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_6, [(X, filters_real, filters_abs, theta_indices, p, grid_size) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped """
## Different amount of rings and scales
def experiment7(X:np.ndarray, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients, [(X, center, grid_scale) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

## Different amount of rings and scales with threshold
def experiment8(X:np.ndarray, threshold: float, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients_with_threshold, [(X, center, grid_scale, threshold) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment4_img(X:np.ndarray, sig:float=20., grid_scale: int=7, grid_radius: int=3) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients_with_img, [(X, p, grid_scale) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment6_img(X:np.ndarray, grid_scale: int=7, grid_radius: int=3, threshold:float = 0.5) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients_with_img_and_threshold, [(X, p, grid_scale, threshold) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment7_img(X:np.ndarray, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients_with_img, [(X, center, grid_scale) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment8_img(X:np.ndarray, threshold: float, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients_with_img_and_threshold, [(X, center, grid_scale, threshold) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

    filters_abs, filters, theta_indices = filter_bank_img_neg(27, sig, 2 * 4, 2)

    manifold = []
    thetas = np.hstack([[np.NaN], np.arange(0, 2 * np.pi, np.pi / 3)])

    with multiprocessing.Pool() as p:
        manifold += p.starmap(calc_5, [(X, filters, filters_abs, theta_indices, t, 0.) for t in thetas])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped
def experiment4_img_neg(X:np.ndarray, sig:float=20., grid_scale: int=7, grid_radius: int=3) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients_with_img_and_neg, [(X, p, grid_scale) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment6_img_neg(X:np.ndarray, grid_scale: int=7, grid_radius: int=3, threshold:float = 0.5) -> np.ndarray:
    manifold = []
    centers = np.rint(grid_scale*hexagon_grid(grid_radius))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(create_coefficients_with_img_and_threshold_and_neg, [(X, p, grid_scale, threshold) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment7_img_neg(X:np.ndarray, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients_with_img_and_neg, [(X, center, grid_scale) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment8_img_neg(X:np.ndarray, threshold: float, grids: list[tuple[int, int]]=[(3, 3), (2, 5), (1, 7), (1, 9)]) -> np.ndarray:
    manifold = []

    with multiprocessing.Pool() as p:
        manifold += p.starmap(create_coefficients_with_img_and_threshold_and_neg, [(X, center, grid_scale, threshold) for grid_radius, grid_scale in grids for center in grid_scale * hexagon_grid(grid_radius)])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped