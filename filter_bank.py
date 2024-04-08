import numpy as np
from gaussian import gaussian


def filter_bank(N: int, sigma: float, r_end: int, r_delta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    n = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n = np.array([n, n.T])

    theta_delta = np.arccos(1. - np.square(r_delta) / (2. * np.square(r_end)))

    filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, np.array([0., 0.]), sigma, True))))
    filters_real = (filter.real / np.sqrt(np.sum(np.square(filter.real)))).reshape(1, N, N).copy()
    filters_imag = np.zeros((1, N, N))
    filters_abs = filter.real.reshape(1, N, N).copy()
    theta_indices = [[0], [0], [0], [0]]
    for theta in np.arange(0., np.pi, theta_delta):
        for r in np.arange(r_delta, r_end + r_delta, r_delta):
            filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))
            filters_real = np.vstack((filters_real, (filter.real / np.sqrt(np.sum(np.square(filter.real)))).reshape(1, N, N)))
            filters_imag = np.vstack((filters_imag, (filter.imag / np.sqrt(np.sum(np.square(filter.imag)))).reshape(1, N, N)))
            filters_abs = np.vstack((filters_abs, (np.abs(filter)).reshape(1, N, N)))
            if theta < np.pi / 8 or theta >= 7 * np.pi / 8:
                theta_indices[0].append(len(filters_real) - 1)
            elif theta >= np.pi / 8 and theta < 3 * np.pi / 8:
                theta_indices[1].append(len(filters_real) - 1)
            elif theta >= 3 * np.pi / 8 and theta < 5 * np.pi / 8:
                theta_indices[2].append(len(filters_real) - 1)
            elif theta >= 5 * np.pi / 8 and theta < 7 * np.pi / 8:
                theta_indices[3].append(len(filters_real) - 1)
    
    return filters_real.astype(np.float32), filters_imag.astype(np.float32), filters_abs.astype(np.float32), theta_indices