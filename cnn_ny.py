import numpy as np
from mnist import get_mnist
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from filter_bank import filter_bank
from sklearn.linear_model import LogisticRegression
import concurrent.futures
from hexagon_grid import hexagon_grid

plt.xticks(())
plt.yticks(())
plt.tight_layout(pad=0.0)

[data_train, labels_train, data_test, labels_test] = get_mnist()
# data_train -= 0.5
data_train = data_train[0:10000, 1:, 1:]
data_test = data_test[:, 1:, 1:]
labels_train = labels_train[0:10000]

# model = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=10000)
# model.fit(data_train.reshape(-1, 27 * 27), labels_train)
# print(model.score(data_test.reshape(-1, 27 * 27), labels_test))

# # embedding = TSNE(n_components=2, random_state=0).fit_transform(data_train.reshape(-1, 27 * 27))
# # np.save('cnn1.npy', embedding)
# embedding = np.load('cnn1.npy')
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=plt.cm.jet, vmin=0, vmax=9, alpha=0.4, linewidths=0)
# colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
# colorbar.set_ticks(np.arange(10))
# colorbar.solids.set_alpha(1.)
# plt.show()

filters_real, filters_imag, filters_abs, theta_indices = filter_bank(27, 20., 2 * 4, 2)

def get_coefficients(data: np.ndarray, center: np.ndarray):

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

def get_coefficients(data: np.ndarray, center: np.ndarray):
    coefficients = np.ones((data.shape[0], len(filters_real))) * -1
    for image_i, image in enumerate(data):
        for filter_i, _ in enumerate(filters_real):
            for dy in np.arange(-3, 4) + center[0]:
                for dx in np.arange(-3, 4) + center[1]:
                    image_masked = np.ones(image.shape)
                    image_masked = np.roll(image, (int(dy), -int(dx)), axis=(0, 1)) * filters_abs[filter_i]
                    image_masked /= np.sqrt(np.sum(np.square(image_masked)))
                    coefficients[image_i, filter_i] = np.max([coefficients[image_i, filter_i], np.sum(image_masked * filters_real[filter_i])])
    coefficients = np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1),
                            coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                            coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1),
                            coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))
    return coefficients


def get_coefficients(center: np.ndarray):

    coefficients = np.zeros((data_train.shape[0], filters_real.shape[0]))

    i, j = np.indices((data_train.shape[0], filters_real.shape[0]))

    for dy in np.arange(-3, 4):
        for dx in np.arange(-3, 4):
            image_masked = np.roll(data_train, np.rint(center + [-dx, dy]).astype(int), axis=(2, 1))[i] * filters_abs[j]
            image_masked /= np.sqrt(np.sum(np.square(image_masked), axis=(2, 3))).reshape(-1, filters_real.shape[0], 1, 1)
            coefficients = np.max([coefficients, np.sum(image_masked * filters_real[j], axis=(2, 3))], axis=0)

    return np.hstack((coefficients[:, theta_indices[0]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[1]].max(axis=1).reshape(-1, 1),
                      coefficients[:, theta_indices[2]].max(axis=1).reshape(-1, 1), coefficients[:, theta_indices[3]].max(axis=1).reshape(-1, 1)))


for center_i, center in enumerate(3 * hexagon_grid(2)):
    np.save(f'cnn_coeff{center_i}.npy', get_coefficients(center))

coefficients = np.hstack([np.load(f'cnn_coeff{i}.npy') for i in range(0, 19)])

# coefficients = np.where(coefficients > 0.5, coefficients, 0.)

# model = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=100000)
# model.fit(coefficients, labels_train)
# print(model.score(coefficients, labels_train))
# exit()

embedding = TSNE(n_components=2, random_state=0).fit_transform(coefficients)
np.save('cnn2.npy', embedding)
embedding = np.load('cnn2.npy')

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=plt.cm.jet, vmin=0, vmax=9, alpha=0.5, linewidths=0)
colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
colorbar.set_ticks(np.arange(10))
colorbar.solids.set_alpha(1.)
plt.show()
