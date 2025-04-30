def split_data(df, split_ratio=None):
    """
    Split the data into train, validation, and test sets.

    Args:
        df (pandas.DataFrame): The dataframe to split.
        split_ratio (list, optional): The ratio for train/val/test.
                                    Defaults to [0.7, 0.15, 0.15].

    Returns:
        tuple: Train, validation, and test dataframes.
    """
    split_ratio = split_ratio or [0.7, 0.15, 0.15]
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"

    n = len(df)
    train_idx = int(n * split_ratio[0])
    val_idx = train_idx + int(n * split_ratio[1])

    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    return train_df, val_df, test_df
