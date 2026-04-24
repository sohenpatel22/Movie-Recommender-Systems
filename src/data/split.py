def temporal_split(ratings_df):
    """
    Time-based train/validation/test split.
    Train: up to 1998-02
    Validation: 1998-03
    Test: 1998-04
    """
    train_df = ratings_df[ratings_df["rating_month"] <= "1998-02"].copy()
    val_df = ratings_df[ratings_df["rating_month"] == "1998-03"].copy()
    test_df = ratings_df[ratings_df["rating_month"] == "1998-04"].copy()

    return train_df, val_df, test_df