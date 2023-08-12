import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_dataset(path: str, label_col: str) -> tuple:
    """Read dataset from csv file

    Parameters
    ----------
    *path: str - dataset path
    *label_col: str - label column name

    Returns
    -------
    row_count: int - row count
    col_count: int - column count
    classes: list - classes
    features: list - features
    df: pd.DataFrame - dataset
    """
    df = pd.read_csv(path)
    row_count = df.shape[0]
    col_count = df.shape[1]
    classes = df[label_col].unique().astype(str).tolist()
    features = df.columns.tolist()
    return (row_count, col_count, classes, features, df)


def split_dataset(
    df: pd.DataFrame,
    label_col: str,
    is_shuffle: bool = True,
    val_size: float = 0.2,
    test_size=0.5,
    normalization: str = "z_score",
    sample_type: str = None,
) -> tuple:
    """Split dataset into train, val, test

    Parameters
    ----------
    *df: pd.DataFrame - dataset
    *label_col: str - label column name
    is_shuffle: bool - shuffle dataset
    val_size: float - validation set size, default 0.2
    test_size: float - test set size, default 0.5
    normalization: str - normalization type, default `z_score`, include `min_max`, `z_score`
    sample_type: str - sample type, default None, include `under`, `over`, `mixed`

    Returns
    -------
    x_train: pd.DataFrame - train set
    x_val: pd.DataFrame - validation set
    x_test: pd.DataFrame - test set
    y_train: pd.Series - train set label
    y_val: pd.Series - validation set label
    y_test: pd.Series - test set label
    """
    # shuffle data
    if is_shuffle:
        df = shuffle(df)
    # split x, y
    x = df.drop(label_col, axis=1)
    y = df[label_col]
    # classes
    classes = y.unique().tolist()
    # normalization x
    if normalization == "min_max":
        x = (x - x.min()) / (x.max() - x.min())
    if normalization == "z_score":
        x = (x - x.mean()) / x.std()
    # split trian, test, val
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, stratify=y, random_state=0
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=test_size, stratify=y_val, random_state=0
    )
    # sample
    if sample_type == "under":
        under_sampler = RandomUnderSampler(random_state=0)
        x_train, y_train = under_sampler.fit_resample(x_train, y_train)
    elif sample_type == "over":
        smote = SMOTE(random_state=0)
        x_train, y_train = smote.fit_resample(x_train, y_train)
    elif sample_type == "mixed":
        smote_enn = SMOTEENN(random_state=0)
        x_train, y_train = smote_enn.fit_resample(x_train, y_train)
    return (x_train, x_val, x_test, y_train, y_val, y_test)
