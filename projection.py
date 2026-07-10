"""Receives the reductions and datasets to perform the projection"""

import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as tree

from feature_selection import *
from train_test import create_folder, save, train_test, unique_train_test

# classifiers
knn_reduction = knn(n_neighbors=1)
gnb_reduction = gnb()
tree_reduction = tree()
lda_reduction = lda()

np.int = int  # Re-defines the missing alias dynamically


def project_datasets(bases, nomes_reducao, numero_repeticoes):

    # for each dataset
    for index in range(len(bases)):
        # dataset data
        X = bases[index][0]
        y = bases[index][1]
        base_name = bases[index][2]
        column_names = bases[index][3]
        # print("Dataset", len(bases[indice]))

        # splitting the n iterations, to be saved later
        # All tests, iterations, classifiers, and methods must use the same train/test split, at least for each dataset
        df_train_indexes = {}
        df_test_indexes = {}
        folder_indexes = "bases/" + base_name + "/dividir_treino_teste"
        create_folder(folder_indexes)
        if os.path.isfile(
            folder_indexes + "/treino-" + str(numero_repeticoes) + ".csv"
        ) and os.path.isfile(
            folder_indexes + "/teste-" + str(numero_repeticoes) + ".csv"
        ):
            df_train_indexes = pd.read_csv(
                folder_indexes + "/treino-" + str(numero_repeticoes) + ".csv"
            )
            df_test_indexes = pd.read_csv(
                folder_indexes + "/teste-" + str(numero_repeticoes) + ".csv"
            )
        else:
            # instead of splitting the matrix, the row indices will be split (workaround)
            X_indexes = np.arange(len(X))
            for iteration in range(numero_repeticoes):
                # use on row indices, not on data
                try:
                    train_x, test_x, train_y, test_y = train_test_split(
                        X_indexes, y, train_size=0.5, test_size=0.5, stratify=y
                    )
                except ValueError:
                    print("This is related with the sample size, adjust later")
                df_train_indexes[str(iteration + 1)] = train_x
                df_test_indexes[str(iteration + 1)] = test_x

            df_train_indexes = pd.DataFrame(
                df_train_indexes, columns=["index", "value"]
            )
            df_test_indexes = pd.DataFrame(df_test_indexes, columns=["index", "value"])
            df_train_indexes.to_csv(
                folder_indexes + "/treino-" + str(numero_repeticoes) + ".csv"
            )
            df_test_indexes.to_csv(
                folder_indexes + "/teste-" + str(numero_repeticoes) + ".csv"
            )

        # print("DATASET: ",nome_base)
        ## catalogs all unique classes in the dataset
        classes = []  # discovered classes
        # add classes that were not added yet, which might only exist in test data
        for clas in y:
            if clas not in classes:
                classes.append(clas)

        maxim = len(X.T)  # maximum number of features

        # for each dimensionality reduction technique
        for reduction_name in nomes_reducao:  # loops through the keys
            # print("reduction: ",nome_reducao)

            # lists of means for each iteration
            meanTree = np.zeros(maxim)
            meanNB = np.zeros(maxim)
            meanKNN = np.zeros(maxim)
            meanLD = np.zeros(maxim)

            # means of all iterations for each dataset
            # each feature will have a list of means to calculate the standard deviation later
            mean_iteration_tree = []
            mean_iteration_NB = []
            mean_iteration_KNN = []
            mean_iteration_LD = []

            folder_save_pca = "bases/" + base_name + "/autvetores_PCA_iteracoes/"
            folder_reduction_results = "bases/" + base_name + "/reducoes_resultados/"
            create_folder(
                folder_save_pca
            )  # creates the directory where each iteration's PCA will be stored
            create_folder(folder_reduction_results)

            # used to track the average feature count for specific reduction strategies
            count_features_reduction = 0
            for iteration in range(numero_repeticoes):
                print(base_name, reduction_name, iteration)
                train_indexes = df_train_indexes[str(iteration + 1)]
                test_indexes = df_test_indexes[str(iteration + 1)]
                train_x, test_x = X[train_indexes], X[test_indexes]
                train_y, test_y = y[train_indexes], y[test_indexes]

                eigenvectors = None
                eigenvalues = None
                train_redux_x = None
                test_redux_x = None

                ######### PCA ##############################

                # numpy save and load are not functioning as before, so eigenvectors will be recalculated
                folder_iteration_autvet = (
                    folder_save_pca + "autvet-" + str(iteration + 1) + ".csv"
                )
                folder_iteration_autval = (
                    folder_save_pca + "autval-" + str(iteration + 1) + ".csv"
                )
                if not os.path.isfile(
                    folder_iteration_autvet
                ):  # if the PCA has not been calculated yet
                    eigenvectors, eigenvalues = PCA(
                        train_x
                    )  # application of regular reduction
                    np.savetxt(folder_iteration_autvet, eigenvectors, delimiter=",")
                    np.savetxt(folder_iteration_autval, eigenvalues, delimiter=",")
                else:  # if they exist, data is loaded
                    eigenvectors = np.loadtxt(folder_iteration_autvet, delimiter=",")
                    eigenvalues = np.loadtxt(folder_iteration_autval, delimiter=",")

                # calculating eigenvectors and eigenvalues just as before
                eigenvectors, eigenvalues = PCA(
                    train_x
                )  # application of regular reduction

                # PCA Projection
                projection_train_x = train_x @ eigenvectors
                projection_test_x = test_x @ eigenvectors
                ##############################################

                result_tree = []
                result_gnb = []
                result_knn = []
                result_lda = []

                ########## Other Types of Dimensionality Reduction ########
                if "info_gain" in reduction_name:
                    sorted, feature_importances = info_gain(
                        projection_train_x, train_y, column_names
                    )
                    feature_importances.to_csv(
                        folder_reduction_results
                        + "info_gain-"
                        + str(iteration + 1)
                        + ".csv"
                    )
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]
                elif "fishers_score" in reduction_name:
                    # feature indices sorted in descending order according to fisher score
                    sorted, feature_importances = fishers_score(
                        projection_train_x, train_y, column_names
                    )
                    feature_importances.to_csv(
                        folder_reduction_results
                        + "fisher_score-"
                        + str(iteration + 1)
                        + ".csv"
                    )
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]
                elif "MCEPCA" in reduction_name:
                    Sk = MCEPCA(
                        projection_train_x,
                        train_x,
                        maxim,
                        classes,
                        train_y,
                        eigenvalues,
                        eigenvectors,
                    )
                    train_redux_x = train_x @ Sk
                    test_redux_x = test_x @ Sk
                elif "PCA" in reduction_name:
                    train_redux_x = projection_train_x
                    test_redux_x = projection_test_x
                elif "chi2_square" in reduction_name:
                    X_teste_cat = projection_train_x.astype(
                        int
                    )  # converting to categorical
                    sorted = chi2_square(X_teste_cat, train_y, maxim)
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]

                elif "correlation_coefficient" in reduction_name:
                    # limitation: must evaluate all features and test a suitable
                    # correlation coefficient threshold (test multiple values)
                    new_data = pd.DataFrame(
                        data=projection_train_x,
                        columns=[str(coluna) for coluna in range(maxim)],
                    )  # creating a temporary dataframe to calculate the correlation matrix
                    sorted = correlation_coefficient(new_data)
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]
                elif "RFE" in reduction_name:
                    sorted = RFE_linear_regression(projection_train_x, train_y)
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]
                elif "Forward" in reduction_name:
                    sorted = forward_linear_regression(projection_train_x, train_y)
                    train_redux_x = projection_train_x[:, sorted]
                    test_redux_x = projection_test_x[:, sorted]
                elif "variance_threshold" in reduction_name:
                    # unlike options above, variance threshold uses this list as definitive, it does not dynamically slice the size
                    ind_features = variance_threshold(projection_train_x, 0.3)
                    print(ind_features)
                    count_features_reduction += len(ind_features)
                    train_redux_x = projection_train_x[:, ind_features]
                    test_redux_x = projection_test_x[:, ind_features]
                elif "LASSO" in reduction_name:
                    ind_features = LASSO(projection_train_x, train_y)
                    print(ind_features)
                    count_features_reduction += len(ind_features)

                    train_redux_x = projection_train_x[:, ind_features]
                    test_redux_x = projection_test_x[:, ind_features]

                ###############################################################

                # outlier results are calculated differently
                # if "correlation_coefficient" not in nome_reducao:
                ############### Classification of Results ###############################

                if "variance_threshold" in reduction_name or "LASSO" in reduction_name:
                    quant_feat = len(X[0])
                    result_tree = list(
                        np.full(
                            quant_feat,
                            unique_train_test(
                                tree_reduction,
                                train_redux_x,
                                test_redux_x,
                                train_y,
                                test_y,
                            ),
                        )
                    )
                    result_gnb = list(
                        np.full(
                            quant_feat,
                            unique_train_test(
                                gnb_reduction,
                                train_redux_x,
                                test_redux_x,
                                train_y,
                                test_y,
                            ),
                        )
                    )
                    result_knn = list(
                        np.full(
                            quant_feat,
                            unique_train_test(
                                knn_reduction,
                                train_redux_x,
                                test_redux_x,
                                train_y,
                                test_y,
                            ),
                        )
                    )
                    result_lda = list(
                        np.full(
                            quant_feat,
                            unique_train_test(
                                lda_reduction,
                                train_redux_x,
                                test_redux_x,
                                train_y,
                                test_y,
                            ),
                        )
                    )
                else:
                    # print(treino_reduzido.shape)

                    ### reshape the training array for tree execution
                    # tree_reduzido_x = np.array([sample] for sample in treino_reduzido_x)
                    # print("Tree")
                    result_tree = train_test(
                        tree_reduction,
                        train_redux_x,
                        test_redux_x,
                        train_y,
                        test_y,
                    )
                    # print("GNB")
                    result_gnb = train_test(
                        gnb_reduction,
                        train_redux_x,
                        test_redux_x,
                        train_y,
                        test_y,
                    )
                    # print("KNN")
                    result_knn = train_test(
                        knn_reduction,
                        train_redux_x,
                        test_redux_x,
                        train_y,
                        test_y,
                    )
                    # print("LDA")
                    result_lda = train_test(
                        lda_reduction,
                        train_redux_x,
                        test_redux_x,
                        train_y,
                        test_y,
                    )

                ### adding results for each individual feature
                meanTree += result_tree
                meanNB += result_gnb
                meanKNN += result_knn
                meanLD += result_lda

                ### matrices used to compute standard deviation
                mean_iteration_tree.append(result_tree)
                mean_iteration_NB.append(result_gnb)
                mean_iteration_KNN.append(result_knn)
                mean_iteration_LD.append(result_lda)

            # the sum vectors for each feature are divided to obtain the mean performance
            meanTree = meanTree / numero_repeticoes
            meanNB = meanNB / numero_repeticoes
            meanKNN = meanKNN / numero_repeticoes
            meanLD = meanLD / numero_repeticoes

            # standard deviation
            deviation_Tree = []
            deviation_NB = []
            deviation_KNN = []
            deviation_LD = []
            # extracting the standard deviation metrics
            for f in range(maxim):
                deviation_Tree.append(pd.Series(meanTree).std())
                deviation_NB.append(pd.Series(meanNB).std())
                deviation_KNN.append(pd.Series(meanKNN).std())
                deviation_LD.append(pd.Series(meanLD).std())

            # saves execution metrics into .txt files
            save(
                meanTree,
                meanKNN,
                meanNB,
                meanLD,  # classifier metric values
                [
                    "resultados",
                    "repeticoes-" + str(numero_repeticoes),
                    base_name,
                ],  # folder paths
                reduction_name,
                maxim,
            )  # strategy label name

            # saves calculated standard deviations into .txt files
            save(
                deviation_Tree,
                deviation_KNN,
                deviation_NB,
                deviation_LD,  # standard deviations from preceding outputs
                [
                    "resultados",
                    "repeticoes-" + str(numero_repeticoes),
                    base_name,
                ],  # folder paths
                reduction_name + "_desvio_padrao",
                maxim,
            )

            if "variance_threshold" in reduction_name or "LASSO" in reduction_name:
                reductions_file = open(
                    "resultados/repeticoes-"
                    + str(numero_repeticoes)
                    + "/"
                    + base_name
                    + "/quantidade-total-reducoes-"
                    + reduction_name
                    + ".txt",
                    "w",
                )
                reductions_file.writelines(str(count_features_reduction))
                reductions_file.close()
                print(
                    "Average feature count: "
                    + str(int(count_features_reduction / numero_repeticoes))
                )
                print("Original count: " + str(len(X[0])))
