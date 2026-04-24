import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler


def load_movielens_data(dataset_dir):
    """
    Load MovieLens 100K user, rating, genre, and item tables.
    """
    users = pd.read_csv(
        dataset_dir / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    ratings = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    ratings["rating_timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["rating_month"] = ratings["rating_timestamp"].dt.to_period("M")

    genres = pd.read_csv(
        dataset_dir / "u.genre",
        sep="|",
        header=None,
        names=["genre", "genre_id"],
    )

    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        names=[
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
        ] + [str(genre) for genre in genres["genre"].tolist()],
    )

    items = (
        items.drop(columns=["video_release_date", "IMDb_URL"])
        .query("title != 'unknown'")
        .reset_index(drop=True)
    )

    return users, ratings, items, genres


def clean_ratings(ratings_df, items_df):
    """
    Remove duplicate user-title interactions, keeping the latest one.
    """
    items_min = items_df[["movie_id", "title"]]

    ratings_clean = (
        ratings_df.merge(items_min, on="movie_id")
        .sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "title"], keep="last")
        .reset_index(drop=True)
    )

    ratings_clean = ratings_clean[ratings_df.columns]
    return ratings_clean


def zip_transform(zipcodes):
    first_char = zipcodes.iloc[:, 0].astype(str).str[0]
    numeric_mask = first_char.str.isnumeric()
    first = first_char.where(numeric_mask, -1).astype(float)
    return pd.DataFrame({zipcodes.columns[0]: first})


def to_datetime(df):
    col = df.iloc[:, 0]
    return pd.DataFrame({col.name: pd.to_datetime(col, errors="coerce")})


def to_unix(df):
    col = df.iloc[:, 0]
    ts = col.astype("int64") // 1_000_000_000
    return pd.DataFrame({col.name: ts.astype(float)})


def build_user_transformer():
    return ColumnTransformer(
        transformers=[
            ("encode", OrdinalEncoder(), ["gender", "occupation"]),
            ("scale", StandardScaler(), ["age"]),
            ("zip", FunctionTransformer(zip_transform, validate=False), ["zip_code"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_movie_transformer():
    date_pipeline = Pipeline(
        steps=[
            ("to_datetime", FunctionTransformer(to_datetime, validate=False)),
            ("to_unix", FunctionTransformer(to_unix, validate=False)),
            ("normalize", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("normalize_release_date", date_pipeline, ["release_date"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def preprocess_tables(users_df, items_df):
    """
    Transform user and movie metadata.
    """
    users_transformer = build_user_transformer()
    movies_transformer = build_movie_transformer()

    users_processed = users_transformer.fit_transform(users_df)
    items_processed = movies_transformer.fit_transform(items_df)

    return users_processed, items_processed