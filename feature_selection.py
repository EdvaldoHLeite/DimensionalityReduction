import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from numpy import linalg as eigen  # eigenvectors and eigenvalues
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import (  # chi square
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    chi2,
    mutual_info_classif,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def PCA(X):  # returns centered matrix and covariance
    matrix_mean = X.mean(axis=0)

    X = X - matrix_mean
    XT = X.T  # X.transpose() # transpose of X or X.T

    covariance = XT @ X  # np.dot(XT, X) # X.T @ X
    covariance = covariance / len(X)
    eig_val, eig_vet = eigen.eig(covariance)  # (np.float(_) for _ in )

    # converting result from complex to real
    eig_val = np.real(eig_val)
    eig_vet = np.real(eig_vet)

    # aut_val, aut_vet = eigen.eig(covariancia.T) # a matrix and its transpose have the same eigenvectors
    # the eigenvectors are the columns
    save = str(eig_val)
    # ordenado = np.argsort(aut_val)[::-1]
    save2 = str(eig_val)
    if save != save2:
        print(
            "Whoops, the eigenvectors were modified during PCA, they cannot be used in MCEPCA"
        )

    """Ed = aut_vet[:, ordenado[:d]]
    aut_vet = Ed
    aut_val = aut_val[ordenado[:d]]"""
    # return Ed, aut_val, aut_vet

    return eig_vet, eig_val


def info_gain(X, y, dataframe_columns):
    # mutual value between features
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, dataframe_columns)  # information gain

    # feature indices sorted by highest information gain
    sorted = np.argsort(feat_importances)[::-1]
    return list(sorted), feat_importances


def fishers_score(X, y, dataframe_columns):
    ranks = fisher_score.fisher_score(X, y)
    feat_importances = pd.Series(ranks, dataframe_columns)
    sorted = np.argsort(feat_importances[::-1])  # sorted in descending order
    return list(sorted), feat_importances


# selects features by lowest sum of coefficients
def correlation_coefficient(data_frame):
    """correlacao = data_frame.corr()
    a=abs(correlacao[target])
    result=a[a<threshold]
    return list(result.index)"""

    correlation = data_frame.corr()  # correlation matrix
    corr_sum = []  # sums of correlations for each feature
    for c in correlation.columns:
        corr_sum.append(correlation[c].sum())

    corr_sum = np.array(corr_sum)
    order = np.argsort(corr_sum)

    return list(order)


# variation selecting the best group for each quantity of features
def correlation_coefficient_2(data_frame):
    correlation = data_frame.corr()  # correlation matrix

    # combinations for each k quantity of features
    index_groups = []
    # starts with the highest possible value of coefficient sums
    minor_sum = (1.0) * len(correlation)
    # for each quantity of features
    for k in range(len(correlation)):
        # for each feature as target, a different group
        best_group = []
        for ind_feat in range(len(correlation)):
            minor_indexes = np.argsort(correlation[ind_feat])[k + 1]
            sum = minor_indexes.sum()

            if sum < minor_sum:
                minor_sum = sum
                best_group = minor_indexes

        index_groups.append(best_group)

    return index_groups


def select_indexes_chi2(chi2_features):
    # selecting prominent indices
    indexes = []
    indexes_bool = chi2_features.get_support()
    for i in range(len(indexes_bool)):
        if indexes_bool[i]:
            indexes.append(i)
    return indexes


def chi2_square(X, y, k):
    # converting to categorical, handling negative cases
    X_cat = X.astype("int")
    X_cat_pos = np.arange(len(X_cat) * len(X_cat[0])).reshape(len(X_cat), len(X_cat[0]))
    uniques = np.unique(X_cat)  # different integer values
    for unique_ind in range(len(uniques)):
        unique = uniques[unique_ind]
        X_cat_pos[X_cat == unique] = unique_ind + 1

    order = []  # sorted indices
    for ki in range(k):  # as ki increases, a new index emerges
        chi2_features = SelectKBest(chi2, k=ki + 1)
        X_kbest_features = chi2_features.fit_transform(X_cat_pos, y)

        # selecting prominent indices
        indexes = select_indexes_chi2(chi2_features)
        # adds the new index to the sorted indices list
        # print(len(np.setdiff1d(ordem, indices)))
        # ordem.append(np.setdiff1d(ordem, indices))
        for index in indexes:
            if index not in order:
                order.append(index)
                break

    return order


def forward_linear_regression(X, y):
    # dataset = pd.DataFrame(data=y, columns=['target'])

    #### converting y to numeric
    """f=0 # number of a target
    valores = y.unique()
    for val in valores:
        f=f+1
        # replaces each target with a number
        dataset['target'].replace([feature], [f])"""
    # different values of y
    """valores = np.unique(y)
    # exchanges a target for a number
    dataset['target'].replace(list(valores), list(np.arange(len(valores))))
    y = np.array(dataset['target'])"""

    lr = LinearRegression()

    order = []
    # finds features for each quantity
    # then appending them to the sorted list, the first ones that appear
    for k in range(len(X[0])):
        ffs = SequentialFeatureSelector(estimator=lr, k_features=k + 1)
        ffs.fit(X, y)
        indexes = ffs.k_feature_idx_  # list(np.where(rfe.support_ == True)[0])
        # checks if the index does not exist in the sorted list
        for index in indexes:
            if index not in order:
                order.append(index)
                break

    return order


def RFE_linear_regression(X, y):
    dataset = pd.DataFrame(data=y, columns=["target"])

    #### converting y to numeric
    """f=0 # number of a target
    valores = y.unique()
    for val in valores:
        f=f+1
        # replaces each target with a number
        dataset['target'].replace([feature], [f])"""
    # different values of y
    valores = np.unique(y)
    # exchanges a target for a number
    dataset["target"].replace(list(valores), list(np.arange(len(valores))))
    y = np.array(dataset["target"])

    lr = LinearRegression()

    order = []
    # finds features for each quantity
    # then appending them to the sorted list, the first ones that appear
    for k in range(len(X[0])):
        rfe = RFE(estimator=lr, n_features_to_select=k + 1, step=1)
        # print("Y in RFE", y)
        rfe.fit(X, y)
        indexes = list(np.where(rfe.support_ == True)[0])
        # checks if the index does not exist in the sorted list
        for index in indexes:
            if index not in order:
                order.append(index)
                break

    return order


def variance_threshold(X, limiar):
    # threshold being 0: the variance of features must be different from zero
    select_feature = VarianceThreshold(threshold=limiar)

    # when it does not find variance for the passed threshold
    try:
        select_feature.fit(X)
    except ValueError:
        return variance_threshold(X, limiar - 0.1)

    # array containing indices of the selected features
    features = []
    # array containing True if the variance is different
    different_variance = select_feature.get_support()
    # print(select_feature.variances_)

    for i in range(len(X[0])):
        if different_variance[i]:
            features.append(i)

    return features


def LASSO(X, y):
    logistic = LogisticRegression(
        C=1, penalty="l1", solver="liblinear", random_state=7
    ).fit(X, y)
    model = SelectFromModel(logistic, prefit=True)

    # array containing indices of the selected features
    features = []
    # array containing True if the variance is different
    different_variance = model.get_support()

    # model2 = make_pipeline(StandardScaler(), Lasso(alpha=.015))
    """model2 = Lasso(alpha=1)
    model2.fit(X, y)
    a = abs(np.array([x for x in model2.coef_]))
    
    order_features = a.argsort()[(-1)*len(a):][::-1]
    return order_features"""

    # print(variancia_diferente)
    # print(np.std(X, 0) * logistic.coef_)
    for i in range(len(X[0])):
        if different_variance[i]:
            features.append(i)

    return features


def MCEPCA(W, X, k, classes, Y, autovalores, autovetores):
    n = len(X)
    d = len(X[0])

    # mean of features
    W_mean = [[0 for x in range(d)] for y in range(2)]
    for c in range(2):  # classes
        for i in range(d):  # features
            numerator = 0
            denominator = 0
            for j in range(n):  # points
                numerator += W[j][i] * int(
                    Y[j] == classes[c]
                )  # wji is the transpose of wij
                denominator += int(Y[j] == classes[c])
            W_mean[c][i] = numerator / denominator

    # score of features
    score = []
    for i in range(d):
        if autovalores[i] != 0:
            score.append((W_mean[0][i] - W_mean[1][i]) ** 2 / autovalores[i])
        else:
            score.append(0)

    sorted = np.argsort(score)[::-1]  # vector containing sorted score indices
    Sk = autovetores[:, sorted[:k]]  # selects the k columns
    return Sk
