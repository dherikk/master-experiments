import numpy as np
from filter_bank import filter_bank
from hexagonal_grid import hexagonal_grid, hexagonal_grid_length
import mnist
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt


def create_coefficients(data: np.ndarray, center: np.ndarray, grid_scale: int):
    filters_real, filters_imag, filters_abs, theta_indices = filter_bank(27, 20., 2 * 4, 2)

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


def load_coefficients(path: str, grid_radius: int, grid_scale: int):
    grid_length = hexagonal_grid_length(grid_radius)
    return np.hstack([np.load(f'coefficients/{path}/scale{grid_scale}_{i}.npy') for i in range(0, grid_length)])


def save_coefficients(path: str, data: np.ndarray, grid_scale: int, center: np.ndarray, center_i: int):
    folder = f'coefficients/{path}'
    os.makedirs(folder, exist_ok=True)

    file = f'{folder}/scale{grid_scale}_{center_i}.npy'
    np.save(file, create_coefficients(data, center, grid_scale))
    print(f'{file} saved')


if __name__ == '__main__':
    [data_train, _, data_test, _] = [mnist.train_images() / 255.0, mnist.train_labels(), mnist.test_images() / 255.0, mnist.test_labels()]
    data_train = data_train[:, 1:, 1:]
    data_test = data_test[:, 1:, 1:] + np.random.normal(0., 0.5, size=(10000, 27, 27))

    pool = Pool()

    for grid_radius, grid_scale in [(3, 3), (2, 5), (1, 7)]:
        for center_i, center in enumerate(grid_scale * hexagonal_grid(grid_radius)):
            # pool.apply_async(save_coefficients, args=('train', data_train, grid_scale, center, center_i))
            pool.apply_async(save_coefficients, args=('test_normal_noise_0.5', data_test, grid_scale, center, center_i))

    pool.close()
    pool.join()
