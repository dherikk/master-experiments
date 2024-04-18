import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def split_train_two(func, X1, X2, y, test_size=0.25, random_state=1, **kwargs):

    i = int((1 - test_size) * X1.shape[0]) 
    o = np.random.default_rng(seed=random_state).permutation(X1.shape[0])
    X_tr, _ = np.split(np.take(X1,o,axis=0), [i])
    _, X_te = np.split(np.take(X2,o,axis=0), [i])
    y_train, y_test = np.split(np.take(y,o), [i])
    if func == None:
        return X_tr, X_te, y_train, y_test
    X_train = func(X_tr, **kwargs)
    X_test = func(X_te, **kwargs)
    return X_train, X_test, y_train, y_test

def score_clfs_noisy(clfs, noisy_dataset):               
    noisy_scores = []
    for noise_i, clf in enumerate(clfs):
        clf.fit(noisy_dataset[noise_i][0], noisy_dataset[noise_i][2])
        noisy_scores.append(clf.score(noisy_dataset[noise_i][1], noisy_dataset[noise_i][3]))
    return noisy_scores

def eval_clfs(clfs, X_te, y_te, tols):
    clf_scores = [clf.score(X_te, y_te) for clf in clfs]
    max_score = np.max(clf_scores)
    y_pred = clfs[np.argmax(clf_scores)].predict(X_te)
    cf_matrix = confusion_matrix(y_te, y_pred)
    _, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot([str(t) for t in tols], clf_scores, linestyle='--', marker='o', color='b', label='Accuracy score')
    ax[0].set_title("Accuracy scores from tol values")
    ax[0].set_ylim([.7, 1.])
    ax[1].set_xlabel("Score")
    ax[1].set_ylabel("tol")
    sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax[1])
    ax[1].set_xlabel("Predicted label")
    ax[1].set_ylabel("True label")
    ax[1].set_title(f"Confusion matrix from estimator\n with highest accuracy score: {max_score}")
    plt.show()

def eval_clfs_svm(clfs, X_te, y_te, Cs):
    clf_scores = [clf.score(X_te, y_te) for clf in clfs]
    max_score = np.max(clf_scores)
    y_pred = clfs[np.argmax(clf_scores)].predict(X_te)
    cf_matrix = confusion_matrix(y_te, y_pred)
    _, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot([str(t) for t in Cs], clf_scores, linestyle='--', marker='o', color='b', label='Accuracy score')
    ax[0].set_title("Accuracy scores from C values")
    ax[0].set_ylim([.5, 1.])
    ax[1].set_xlabel("Score")
    ax[1].set_ylabel("tol")
    sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax[1])
    ax[1].set_xlabel("Predicted label")
    ax[1].set_ylabel("True label")
    ax[1].set_title(f"Confusion matrix from estimator\n with highest accuracy score: {max_score}")
    plt.show()