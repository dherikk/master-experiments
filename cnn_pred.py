from cnn_parallell import experiment1, experiment3, experiment4
import numpy as np
import seaborn as sns
import mnist
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def score_clf(clf, trainset, trainlabels, testset, testlabels):
    train_score = clf.score(trainset, trainlabels)
    print("Train score:", train_score.round(4))
    test_score = clf.score(testset, testlabels)
    print("Test score:", test_score.round(4))
    y_pred = clf.predict(testset)
    cf_matrix = confusion_matrix(testlabels, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label") 
    plt.show()

def trainer_baseline_logistic(train, labels):
    clf = LogisticRegression(fit_intercept=True,
                             multi_class='multinomial',
                             penalty='l2',
                             solver='saga',
                             max_iter=1000,
                             verbose=2,
                             n_jobs=8,
                             tol=0.01
                            )
    clf.fit(train, labels)
    return clf

def trainer_baseline_svm(train, labels):
    svm = SVC(kernel='rbf', C=10., gamma="auto")
    svm.fit(train, labels)
    return svm

def transformed_data_predict_logistic(train, labels, clf):
    clf = LogisticRegression(fit_intercept=True,
                             multi_class='multinomial',
                             penalty='l2',
                             solver='saga',
                             max_iter=1000,
                             verbose=2,
                             n_jobs=8,
                             tol=0.001
                            )
    transformed_train = experiment1(train)
    clf.fit(transformed_train, labels)
    return clf, transformed_train

def transformed_data_predict_svm(train, labels):
    svm = SVC(kernel='rbf', C=50., gamma="auto")
    transformed_train = experiment1(train)
    svm.fit(transformed_train, labels)
    return svm, transformed_train

def flower_transform_predict_logistic(train, labels):
    clf = LogisticRegression(fit_intercept=True,
                             multi_class='multinomial',
                             penalty='l2',
                             solver='saga',
                             max_iter=1000,
                             verbose=2,
                             n_jobs=8,
                             tol=0.0005
                            )
    transformed_train = experiment3(train)
    clf.fit(transformed_train, labels)
    return clf, transformed_train

def flower_transform_predict_svm(train, labels):
    svm = SVC(kernel='rbf', C=50., gamma="auto")
    transformed_train = experiment3(train)
    svm.fit(transformed_train, labels)
    return svm, transformed_train

def larger_transform_logistic(train, labels):
    clf = LogisticRegression(fit_intercept=True,
                             multi_class='multinomial',
                             penalty='l2',
                             solver='saga',
                             max_iter=1000,
                             verbose=2,
                             n_jobs=8,
                             tol=0.001
                            )
    transformed_train = experiment4(train)
    clf.fit(transformed_train, labels)
    return clf, transformed_train

def larger_transform_svm(train, labels):
    svm = SVC(kernel='rbf', C=100., gamma="auto")
    transformed_train = experiment4(train)
    svm.fit(transformed_train, labels)
    return svm, transformed_train

def trainLRestimate(X_tr, y_tr, tols):
    clfs = [LogisticRegression(fit_intercept=True,
                               multi_class='multinomial',
                               penalty='l2',
                               solver='saga',
                               max_iter=1000,
                               n_jobs=8,
                               tol=tol) for tol in tols]
    for clf in clfs:
        clf.fit(X_tr, y_tr)
    return clfs

def trainSVMestimate(X_tr, y_tr, Cs):
    clfs = [SVC(kernel='rbf', C=C, gamma="auto") for C in Cs]
    for clf in clfs:
        clf.fit(X_tr, y_tr)
    return clfs

def score_clfs_noisy(clfs, noisy_dataset):               
    noisy_scores = []
    for noise_i, clf in enumerate(clfs):
        clf.fit(noisy_dataset[noise_i][0], noisy_dataset[noise_i][2])
        noisy_scores.append(clf.score(noisy_dataset[noise_i][1], noisy_dataset[noise_i][3]))
    return noisy_scores