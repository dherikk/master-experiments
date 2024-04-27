import numpy as np
from components.get_mnist import get_mnist
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from components.gaussian import gaussian

plt.xticks(())
plt.yticks(())
plt.tight_layout(pad=0.0)

[data_train, labels_train, _, _] = get_mnist()
# data_train -= 0.5
data_train = data_train[0:10000, 1:, 1:]
labels_train = labels_train[0:10000]

# # embedding = TSNE(n_components=2, random_state=0).fit_transform(data_train.reshape(-1, 27 * 27))
# # np.save('cnn1.npy', embedding)
# embedding = np.load('cnn1.npy')
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=plt.cm.jet, vmin=0, vmax=9, alpha=0.4, linewidths=0)
# colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
# colorbar.set_ticks(np.arange(10))
# colorbar.solids.set_alpha(1.)
# plt.show()

N = 27
n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
n2 = n1.T
n = np.array([n1, n2])
sigma = 20.
dr = 2.
dtheta = 0.250655662336131
filters_real = []
filters_abs = []
frequency_domain = gaussian(n, np.array([0., 0.]), sigma, True)
filter_real = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain))).real
filters_real.append(filter_real)
filters_abs.append(filter_real)
theta_indices = [[0], [0], [0], [0]]
for theta in np.arange(0., np.pi, dtheta):
    for r in np.arange(dr, dr * 4. + dr, dr):
        filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))
        filters_real.append(filter.real / np.sqrt(np.sum(np.square(filter.real))))
        filters_abs.append(np.abs(filter))
        if theta < np.pi / 8 or theta >= 7 * np.pi / 8:
            theta_indices[0].append(len(filters_real) - 1)
        elif theta >= np.pi / 8 and theta < 3 * np.pi / 8:
            theta_indices[1].append(len(filters_real) - 1)
        elif theta >= 3 * np.pi / 8 and theta < 5 * np.pi / 8:
            theta_indices[2].append(len(filters_real) - 1)
        elif theta >= 5 * np.pi / 8 and theta < 7 * np.pi / 8:
            theta_indices[3].append(len(filters_real) - 1)

for theta_i, theta in enumerate(np.hstack([[np.NaN], np.arange(0, 2 * np.pi, np.pi / 3)])):
    coefficients = np.ones((data_train.shape[0], len(filters_real))) * -1
    for image_i, image in enumerate(data_train):
        for filter_i, filter_real in enumerate(filters_real):
            for dy in np.arange(-3, 4):
                for dx in np.arange(-3, 4):
                    image_masked = np.roll(image, [dy, -dx] if np.isnan(theta) else [int(dy + np.rint(7 * np.sin(theta))), -int(dx + np.rint(7 * np.cos(theta)))], axis=(0, 1)) * filters_abs[filter_i]
                    image_masked /= np.sqrt(np.sum(np.square(image_masked)))
                    coefficients[image_i, filter_i] = np.max([coefficients[image_i, filter_i], np.sum(image_masked * filters_real[filter_i])])
                    # if np.sum(image_masked * filters_real[filter_i]) > 0.9:
                    #     print(image_i)
                    #     plt.imshow(image_masked)
                    #     plt.show()
    coefficients = np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                              coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    np.save(f'cnn_coeff{theta_i}.npy', coefficients)

coefficients = np.hstack([np.load(f'cnn_coeff{i}.npy') for i in range(0, 7)])

# coefficients = np.where(coefficients > 0.5, coefficients, 0.)

embedding = TSNE(n_components=2, random_state=0).fit_transform(coefficients)
np.save('cnn2.npy', embedding)
embedding = np.load('cnn2.npy')

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=plt.cm.jet, vmin=0, vmax=9, alpha=0.5, linewidths=0)
colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
colorbar.set_ticks(np.arange(10))
colorbar.solids.set_alpha(1.)
plt.show()
