import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def sorted_kfold(df: pd.DataFrame, y_col: str, n_splits: int):
    """
    Sorted Stratified Kfold

    :param df: input dataframe
    :param y_col: targe column
    :param n_splits: number of splits of Kfold
    :return: dataframe reordered based on n_splits
    """
    n_samples = df.shape[0]
    current = 0
    total_len = 0
    new_order = list()

    df = df.sort_values(by=y_col).reset_index(drop=True)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
    fold_sizes[:n_samples % n_splits] += 1

    for fold_size in fold_sizes:
        new_order.extend([current + ix + ((n_splits - 1) * ix) for ix in range(fold_size)])
        current += 1

    return df.reindex(new_order).reset_index(drop=True)


if __name__ == '__main__':
    n_splits = 5

    df = pd.DataFrame({"input_column": np.arange(150), 'target': np.random.normal(0, 1, 150)})
    df_stratified = sorted_kfold(df, 'target', n_splits)

    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_index, test_index in kf.split(df_stratified):
        print("TRAIN:", train_index, "TEST:", test_index)
