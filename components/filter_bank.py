import numpy as np
from components.gaussian import gaussian
import matplotlib.pyplot as plt


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

def filter_bank_img(N: int, sigma: float, r_end: int, r_delta: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    n = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n = np.array([n, n.T])

    theta_delta = np.arccos(1. - np.square(r_delta) / (2. * np.square(r_end)))

    filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, np.array([0., 0.]), sigma, True))))
    filters = filter.real.reshape(1, N, N).copy()
    filters_abs = filter.real.reshape(1, N, N).copy()
    theta_indices = [[], [], [], []]
    for theta in np.arange(0., np.pi, theta_delta):
        for r in np.arange(r_delta, r_end + r_delta, r_delta):
            theta_rot = theta + np.pi
            filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))
            filter_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta_rot), np.sin(theta_rot)]), sigma, True))))
            filters = np.vstack((filters, (filter.real / np.sqrt(np.sum(np.square(filter.real)))).reshape(1, N, N)))
            filters = np.vstack((filters, (filter.imag / np.sqrt(np.sum(np.square(filter.imag)))).reshape(1, N, N)))
            filters = np.vstack((filters, (filter_2.imag / np.sqrt(np.sum(np.square(filter_2.imag)))).reshape(1, N, N)))
            filters_abs = np.vstack((filters_abs, (np.abs(filter)).reshape(1, N, N)))
            if theta < np.pi / 8 or theta >= 7 * np.pi / 8:
                theta_indices[0].append(len(filters) - 2)
                theta_indices[0].append(len(filters) - 3)
                theta_indices[0].append(len(filters) - 4)
            elif theta >= np.pi / 8 and theta < 3 * np.pi / 8:
                theta_indices[1].append(len(filters) - 2)
                theta_indices[1].append(len(filters) - 3)
                theta_indices[1].append(len(filters) - 4)
            elif theta >= 3 * np.pi / 8 and theta < 5 * np.pi / 8:
                theta_indices[2].append(len(filters) - 2)
                theta_indices[2].append(len(filters) - 3)
                theta_indices[2].append(len(filters) - 4)
            elif theta >= 5 * np.pi / 8 and theta < 7 * np.pi / 8:
                theta_indices[3].append(len(filters) - 2)
                theta_indices[3].append(len(filters) - 3)
                theta_indices[3].append(len(filters) - 4)

    return filters_abs[1:].astype(np.float32), filters[1:].astype(np.float32), theta_indices

def filter_bank_img_neg(N: int, sigma: float, r_end: int, r_delta: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    n = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n = np.array([n, n.T])

    theta_delta = np.arccos(1. - np.square(r_delta) / (2. * np.square(r_end)))

    filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, np.array([0., 0.]), sigma, True))))
    filters = filter.real.reshape(1, N, N).copy()
    filters_abs = filter.real.reshape(1, N, N).copy()
    theta_indices = [[], [], [], []]
    for theta in np.arange(0., np.pi, theta_delta):
        for r in np.arange(r_delta, r_end + r_delta, r_delta):
            theta_rot = theta + np.pi
            filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))
            filter_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta_rot), np.sin(theta_rot)]), sigma, True))))
            filters = np.vstack((filters, (filter.real / np.sqrt(np.sum(np.square(filter.real)))).reshape(1, N, N)))
            filters = np.vstack((filters, (-filter.real / np.sqrt(np.sum(np.square(filter.real)))).reshape(1, N, N)))
            filters = np.vstack((filters, (filter.imag / np.sqrt(np.sum(np.square(filter.imag)))).reshape(1, N, N)))
            filters = np.vstack((filters, (filter_2.imag / np.sqrt(np.sum(np.square(filter_2.imag)))).reshape(1, N, N)))
            filters_abs = np.vstack((filters_abs, (np.abs(filter)).reshape(1, N, N)))
            if theta < np.pi / 8 or theta >= 7 * np.pi / 8:
                theta_indices[0].append(len(filters) - 2)
                theta_indices[0].append(len(filters) - 3)
                theta_indices[0].append(len(filters) - 4)
                theta_indices[0].append(len(filters) - 5)
            elif theta >= np.pi / 8 and theta < 3 * np.pi / 8:
                theta_indices[1].append(len(filters) - 2)
                theta_indices[1].append(len(filters) - 3)
                theta_indices[1].append(len(filters) - 4)
                theta_indices[1].append(len(filters) - 5)
            elif theta >= 3 * np.pi / 8 and theta < 5 * np.pi / 8:
                theta_indices[2].append(len(filters) - 2)
                theta_indices[2].append(len(filters) - 3)
                theta_indices[2].append(len(filters) - 4)
                theta_indices[2].append(len(filters) - 5)
            elif theta >= 5 * np.pi / 8 and theta < 7 * np.pi / 8:
                theta_indices[3].append(len(filters) - 2)
                theta_indices[3].append(len(filters) - 3)
                theta_indices[3].append(len(filters) - 4)
                theta_indices[3].append(len(filters) - 5)

    return filters_abs[1:].astype(np.float32), filters[1:].astype(np.float32), theta_indices

# main function
if __name__ == '__main__':
    filters_real, filters, theta_indices = filter_bank_img(27, 20., 8, 2)
    print(filters_real.shape, filters.shape)
    print(156//3)
    print(len(theta_indices[0]), len(theta_indices[1]), len(theta_indices[2]), len(theta_indices[3]))
    print(theta_indices[0])
    print(theta_indices[1])
    print(theta_indices[2])
    print(theta_indices[3])
