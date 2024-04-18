import numpy as np
import matplotlib.pyplot as plt
from gaussian import gaussian
from typing import Union, Any, Tuple
from hexagon_grid import hexagon_grid
from filter_bank import filter_bank
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

def experiment1(X:np.ndarray, sig:float=20.) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)

    manifold = []
    thetas = np.hstack([[np.NaN], np.arange(0, 2 * np.pi, np.pi / 3)])

    with multiprocessing.Pool() as p:
        manifold += p.starmap(calc, [(X, filters_real, filters_abs, theta_indices, t) for t in thetas])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment2(X:np.ndarray, sig:float=20.) -> np.ndarray:

    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(8*hexagon_grid(2))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_4, [(X, filters_real, filters_abs, theta_indices, p) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment3(X:np.ndarray, sig:float=20.) -> np.ndarray:
    N = 27
    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])
    sigma = sig
    dr = 2.
    dtheta = 0.250655662336131

    filters_real_ = []
    filters_abs_ = []
    all_indices = [[[], [], []], [[0], [], []], [[0], [], []], [[0], [], []]]

    frequency_domain = gaussian(n, np.array([0., 0.]), sigma, True)
    filter_real = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain))).real
    filters_real_.append(filter_real)
    filters_abs_.append(filter_real)

    for theta in np.arange(0., np.pi, dtheta):
        for r_idx, r in enumerate(np.arange(dr + dr , dr * 4. + dr, dr)):
            filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))
            filters_real_.append(filter.real / np.sqrt(np.sum(np.square(filter.real))))
            filters_abs_.append(np.abs(filter))
            if theta < np.pi / 8 or theta >= 7 * np.pi / 8:
                all_indices[0][r_idx].append(len(filters_real_) - 1)
            elif theta >= np.pi / 8 and theta < 3 * np.pi / 8:
                all_indices[1][r_idx].append(len(filters_real_) - 1)
            elif theta >= 3 * np.pi / 8 and theta < 5 * np.pi / 8:
                all_indices[2][r_idx].append(len(filters_real_) - 1)
            elif theta >= 5 * np.pi / 8 and theta < 7 * np.pi / 8:
                all_indices[3][r_idx].append(len(filters_real_) - 1)

    theta_indices = [[item for items in all_indices[i] for item in items] for i in range(4)]
    circle_i = [[i[ring] for i in all_indices] for ring in range(3)]
    circle_indices = [[x for xs in lst for x in xs] for lst in circle_i]

    filters_real = np.array(filters_real_)
    filters_abs = np.array(filters_abs_)

    manifold = []
    visited = []
    all_centers = []

    for theta in np.arange(0, 2 * np.pi, np.pi / 3):
        if ((8 * np.sin(theta), -8 * np.cos(theta)) not in visited and not np.isnan(theta)):
            visited.append((8 * np.sin(theta), -8 * np.cos(theta)))

    for tt in np.arange(0, 2 * np.pi, np.pi / 3):
        for visit in visited.copy():
            if ((visit[0] + (8 * np.sin(tt)), visit[1] + (8 * np.cos(tt))) not in visited):
                point = (visit[0] + (8 * np.sin(tt)), visit[1] - (8 * np.cos(tt)))
                all_centers.append(point)

    centers = set([(np.rint(i[0]), np.rint(i[1])) for i in all_centers])
    centers = np.array([list(i) for i in centers])
 
    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_3, [(X, filters_real, filters_abs, theta_indices, circle_indices, p) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment4(X:np.ndarray, sig:float=20., length: int=6, rings: int=3) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(length*hexagon_grid(rings))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_4, [(X, filters_real, filters_abs, theta_indices, p) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment6(X:np.ndarray, sig:float=20., length: int=6, rings: int=3, threshhold:float = 0.5) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(length*hexagon_grid(rings))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_5, [(X, filters_real, filters_abs, theta_indices, p, threshhold) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped

def experiment7(X:np.ndarray, sig:float=20., length: int=6, rings: int=3, grid_size=3) -> np.ndarray:
    filters_real, _, filters_abs, theta_indices = filter_bank(27, sig, 2 * 4, 2)
    manifold = []
    centers = np.rint(length*hexagon_grid(rings))

    with multiprocessing.Pool(len(centers)) as p:
        manifold += p.starmap(calc_6, [(X, filters_real, filters_abs, theta_indices, p, grid_size) for p in centers])

    manifold_reshaped = np.hstack([i for i in manifold])

    return manifold_reshaped