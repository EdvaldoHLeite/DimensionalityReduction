import pandas as pd

from load_datasets import (
    banknote,
    climate,
    debrecen,
    letter,
    mice,
    obs_network,
    occupancy,
    pima,
    spambase,
    user_knowledge,
    vcolumn,
    waveform,
    wdbc,
    wine_quality_red,
    wine_quality_white,
)

opt = 3

bases3 = [
    letter()[2],
    user_knowledge()[2],
    mice()[2],
    wine_quality_red()[2],
    wine_quality_white()[2],
    waveform()[2],
]
reductions3 = ["PCA", "LASSO", "variance_threshold"]

bases2 = [
    banknote()[2],
    climate()[2],
    debrecen()[2],
    occupancy()[2],
    pima()[2],
    vcolumn()[2],
    wdbc()[2],
]
reductions2 = ["MCEPCA", "LASSO", "variance_threshold"]

bases = None
reductions = None
classifiers = ["tree", "knn", "gnb", "lda"]
if opt == 2:
    bases = bases2
    reductions = reductions2
else:
    bases = bases3
    reductions = reductions3

reductions.reverse()
for b in bases:
    for r in reductions:
        file_name = "resultados/repeticoes-100/" + b + "/" + r + ".txt"
        f = pd.read_csv(file_name, sep=",")

        print(b, r)
        print(f[classifiers].mean())
        print("\n")
