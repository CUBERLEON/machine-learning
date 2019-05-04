from typing import List

import pandas as pd


def prepair_categorical_features(df: pd.DataFrame, columns: List[str]):
    for col in columns:
        print(f"{col} values = ", sorted(df[col].unique()))

    dummy_cols = [pd.get_dummies(df[col], prefix=col) for col in columns]
    result = pd.concat([df, *dummy_cols], axis=1)
    result = result.drop(columns=columns)
    return result
