from pathlib import Path
from loguru import logger
from loguru import logger
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import sys

sys.path.insert(0, "..")
root = Path("..")


def create_labeled_data():
    """This function creates the labeled dataset by combining the two files based on the spotify ID"""

    ## Define filepaths to directories and files
    data_dir = root / "src" / "data" / "raw"
    result_dir = root / "src" / "data" / "processed"

    meta_file = data_dir / "spotify_id_metadata.csv"
    trackid_file = data_dir / "tracks_with_spotifyid.csv"
    label_file = data_dir / "songlist.csv"

    ## Where resulting dataset will be written to as csv
    result_file = result_dir / "labeled_data.csv"

    ## read in source files
    df_meta = pd.read_csv(meta_file)
    df_trackid = pd.read_csv(trackid_file)
    df_label = pd.read_csv(label_file, usecols=["track_id", "duration", "genre"])

    ## Combine file with metadata on tracks with the original songlist ID
    ## and drop some irrelevant columns
    df = pd.merge(df_meta, df_trackid, left_on="id", right_on="spotify_id", how="inner")

    columns_to_drop = [
        "track_href",
        "search_artist",
        "search_track",
        "id",
        "Unnamed: 0",
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    ## Merge the music genre labels with the tracks including metadata.
    df_out = pd.merge(
        df, df_label, left_on="track_id", right_on="track_id", how="inner"
    )

    df_out.to_csv(result_file, index=False)

    return df_out


def create_train_test_validation(
    train_ratio: float = 0.7, test_ratio: float = 0.15, validation_ratio: float = 0.15
):
    """This function will create a train, test and validation set including transformation and preprocessing steps like
    onehot encoding and scaling the numeric values"""

    data_dir: Path = root / "processed"
    input_file: str = "labeled_data.csv"
    file_path: Path = data_dir / input_file

    if file_path.exists():
        logger.info(
            f"found file {input_file}, procceed with creating train, test and validation sets"
        )
        try:
            df = pd.read_csv(file_path)
        except:
            logger.info(f"an error occured while trying to pd.read_csv {file_path}")

    else:
        logger.info("Labeled data not found, creating a new file.")
        create_labeled_data()

    ## remove these columns
    columns_to_drop = [
        "type",
        "uri",
        "analysis_url",
        "time_signature",
        "track_id",
        "response_artist",
        "response_track",
        "spotify_id",
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    ## make these columns categorical variables

    for col in ["key", "mode", "genre"]:
        df[col] = df[col].astype("category")

    ##initialize encoders

    ohe = OneHotEncoder(sparse=False)
    scaler = StandardScaler()
    oe = OrdinalEncoder()

    ## Transform genre to ordinal coding
    ordinal_columns = oe.fit_transform(df["genre"].to_numpy().reshape(-1, 1))

    ## apply OneHot encoding to categorical variables
    columns_to_onehot = ["key"]
    onehot_columns = ohe.fit_transform(df[columns_to_onehot])

    ## Apply StandardScaler to numeric values
    columns_to_scale = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    scaled_columns = scaler.fit_transform(df[columns_to_scale])

    ## Not forget about the already binary feature 'mode'

    binary_columns = df["mode"].to_numpy().reshape(-1, 1)

    dataset = np.concatenate(
        [onehot_columns, scaled_columns, binary_columns, ordinal_columns], axis=1
    )

    if train_ratio + test_ratio + validation_ratio != 1:
        logger.info(
            "Inputted ratio do not count up to 1. Make sure all ratio's add up to 1"
        )
    else:

        x = dataset[:, :-1]
        y = dataset[:, -1]

        train_n = len(dataset) * train_ratio
        test_n = len(dataset) * test_ratio
        valid_n = len(dataset) * validation_ratio
        x_train, x_test, x_valid = x[:train_n], x[train_n:test_n], x[test_n:valid_n]
        y_train, y_test, y_valid = y[:train_n], y[train_n:test_n], y[test_n:valid_n]

    return x_train, y_train, x_test, y_test, x_valid, y_valid
