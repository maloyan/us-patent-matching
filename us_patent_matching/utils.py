import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(data, num_splits):
    data["fold"] = -1
    data.loc[:, "bins"] = pd.cut(
        data["score"], bins=5, labels=False
    )
    
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'fold'] = f
    data = data.drop("bins", axis=1)

    return data

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }

def mapping_codes(row):
    return f"{row['section']}{int(row['class']):02}"