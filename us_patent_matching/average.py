   
import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
df_gt = pd.read_csv("/root/us_patent_matching/input/train.csv")[['id', 'score']]
model_files = [
    'bert',
    'debertav3',
    'xlm',
    'roberta'
]
target_cols = ['preds']
df_oof = pd.concat(
    [
        pd.read_csv(f"/root/us_patent_matching/submissions/oof_{i}.csv")[
            target_cols
        ].add_suffix(f"_{i}")
        for i in model_files
    ],
    axis=1,
)
df_oof['id'] = pd.read_csv(f"/root/us_patent_matching/submissions/oof_{model_files[0]}.csv")['id']
df = pd.merge(df_gt, df_oof, on='id')
# clf = LogisticRegression(
#     C=1, solver="newton-cg", penalty="l2", n_jobs=-1, max_iter=100
clf = Ridge().fit(df[[i for i in df.columns if i.startswith('preds')]], df['score'])

print([i for i in df.columns if i.startswith('preds')], clf.coef_)