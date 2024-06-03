import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from mnist import get_mnist
import mnist
from coef import load_coefficients
from enum import Enum

plt.xticks(())
plt.yticks(())
plt.tight_layout(pad=0.0)



train_size = 60000


def raw_tsne(threshold: None | float):
    [data_train, labels_train] = [mnist.train_images()/255.0, mnist.train_labels()]
    data_train = data_train[0:train_size, 1:, 1:]
    labels_train = labels_train[0:train_size]

    if threshold:
        data_train = np.where(data_train > threshold, data_train, 0.)

    embedding = TSNE(n_components=2, random_state=0).fit_transform(data_train.reshape(-1, 27 * 27))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=cm.get_cmap('jet'), vmin=0, vmax=9, alpha=0.4, linewidths=0)
    colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    colorbar.set_ticks(np.arange(10))
    colorbar.solids.set_alpha(1.)
    plt.show()


def coefficient_tsne(threshold: None | float):
    [_, labels_train, _, _] = [None, mnist.train_labels(), None, None]
    labels_train = labels_train[0:train_size]

    coefficients_train = np.empty((train_size, 0))
    for grid_radius, grid_scale in [(3, 3), (2, 5), (1, 7)]:
        coefficients_train = np.hstack((coefficients_train, load_coefficients('train', grid_radius, grid_scale)[0:train_size]))

    if threshold:
        coefficients_train = np.where(coefficients_train > threshold, coefficients_train, 0.)

    embedding = TSNE(n_components=2, random_state=0).fit_transform(coefficients_train)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_train, cmap=cm.get_cmap('jet'), vmin=0, vmax=9, alpha=0.5, linewidths=0)
    colorbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    colorbar.set_ticks(np.arange(10))
    colorbar.solids.set_alpha(1.)
    plt.show()


class Classifier(Enum):
    SUPPORT_VECTOR = 1
    LOGISTIC_REGRESSION = 2
    K_NEAREST_NEIGHBOURS = 3


def get_model(classifier: Classifier):
    if classifier == Classifier.SUPPORT_VECTOR:
        return SVC(random_state=0, kernel='rbf', C=10., gamma="auto")
    elif classifier == Classifier.LOGISTIC_REGRESSION:
        return LogisticRegression(fit_intercept=True,
                                  max_iter=1000,
                                  penalty='l2',
                                  multi_class='multinomial',
                                  solver='saga',
                                  n_jobs=16,
                                  tol=0.001)
    elif classifier == Classifier.K_NEAREST_NEIGHBOURS:
        return KNeighborsClassifier(n_neighbors=5, p=2)


def raw_score(noise_scale: None | float, threshold: None | float, classifier: Classifier):
    [data_train, labels_train, data_test, labels_test] = [mnist.train_images()/255.0, mnist.train_labels(), mnist.test_images()/255.0, mnist.test_labels()]
    data_train = data_train[0:train_size, 1:, 1:]
    data_test = data_test[:, 1:, 1:] + (np.random.normal(0., noise_scale, size=(10000, 27, 27)) if noise_scale else 0.) ### train_size was 10000
    labels_train = labels_train[0:train_size]

    if threshold:
        data_train = np.where(data_train > threshold, data_train, 0.)
        data_test = np.where(data_test > threshold, data_test, 0.)

    model = get_model(classifier)
    model.fit(data_train.reshape(-1, 27 * 27), labels_train)
    print("Done fitting, now scoring")
    return model.score(data_test.reshape(-1, 27 * 27), labels_test)


def coefficient_score(noise_scale: None | float, threshold: None | float, classifier: Classifier):
    [_, labels_train, _, labels_test] = [None, mnist.train_labels(), None, mnist.test_labels()]
    labels_train = labels_train[0:train_size]

    coefficients_train = np.empty((train_size, 0))
    coefficients_test = np.empty((10000, 0))
    for grid_radius, grid_scale in [(3, 3), (2, 5), (1, 7)]:
        coefficients_train = np.hstack((coefficients_train, load_coefficients('train', grid_radius, grid_scale)[0:train_size]))
        coefficients_test = np.hstack(
            (coefficients_test, load_coefficients(f'test_normal_noise_{noise_scale}' if noise_scale else 'test', grid_radius, grid_scale)))

    if threshold:
        coefficients_train = np.where(coefficients_train > threshold, coefficients_train, 0.)
        coefficients_test = np.where(coefficients_test > threshold, coefficients_test, 0.)

    model = get_model(classifier)
    model.fit(coefficients_train, labels_train)
    return model.score(coefficients_test, labels_test)  # SVC, grid 4, scale 3: 0:10000: 0.9622, 0:20000: 0.9684, 0:60000: 0.9787


if __name__ == '__main__':
    classifier = Classifier.K_NEAREST_NEIGHBOURS
    print("0.7 terskling, KNN, (3,3), (2,5), (1,7)")
    print(coefficient_score(noise_scale=0.5, threshold=None, classifier=classifier))
