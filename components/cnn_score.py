import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def score_clfs_noisy(clfs, X_train, y_train, X_test, y_test):
    scores = []
    for i, clf in enumerate(clfs):
        clf.fit(X_train, y_train)  
        scores.append(clf.score(X_test[i], y_test))
    return scores


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