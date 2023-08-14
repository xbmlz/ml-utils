import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_dataset(
    csv_file: str,
    label_col: str,
    is_shuffle: bool = True,
    normalization: str = "z_score",
    val_size: float = 0.2,
    test_size: float = 0.5,
    sample_type: str = None,
) -> tuple:
    """Read dataset from csv file, split dataset into train, val, test

    Parameters
    ----------
    csv_file: str
        The path of csv file
    label_col: str
        Label column name
    is_shuffle: bool, default True
        Shuffle dataset
    normalization: str, default `z_score`, include `min_max`, `z_score`
        Normalization type
    val_size: float, default 0.2
        Validation set size
    test_size: float, default 0.5
        Test set size
    sample_type: str, default None, include `under`, `over`, `mixed`
        Sample type

    Returns
    -------
    df: pd.DataFrame - dataset
    x_train: pd.DataFrame - train set
    x_val: pd.DataFrame - validation set
    x_test: pd.DataFrame - test set
    y_train: pd.Series - train set label
    y_val: pd.Series - validation set label
    y_test: pd.Series - test set label

    Examples
    --------
    >>> from ml_utils.dataset import read_dataset
    >>> df, x_train, x_val, x_test, y_train, y_val, y_test = read_dataset(
    ...     csv_file="dataset.csv",
    ...     label_col="label",
    ...     is_shuffle=True,
    ...     normalization="z_score",
    ...     val_size=0.2,
    ...     test_size=0.5,
    ...     sample_type="under",
    ... )
    """
    df = pd.read_csv(csv_file)
    # shuffle data
    if is_shuffle:
        df = shuffle(df)
    # split x, y
    x = df.drop(label_col, axis=1)
    y = df[label_col]
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
    return df, x_train, x_val, x_test, y_train, y_val, y_test
