from pathlib import Path
from loguru import logger
from loguru import logger
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
import sys

sys.path.insert(0,'..')
root = Path('..')


def create_labeled_data():
    """This function creates the labeled dataset by combining the two files based on the spotify ID"""
    
    
    ## Define filepaths to directories and files
    data_dir = root / 'src' / 'data' / 'raw'
    result_dir = root / 'src' / 'data' / 'processed'

    meta_file = data_dir / 'spotify_id_metadata.csv'
    trackid_file = data_dir / 'tracks_with_spotifyid.csv'
    label_file = data_dir / 'songlist.csv'

    ## Where resulting dataset will be written to as csv
    result_file = result_dir / 'labeled_data.csv'

    ## read in source files
    df_meta = pd.read_csv(meta_file)
    df_trackid = pd.read_csv(trackid_file)
    df_label = pd.read_csv(label_file,usecols = ['track_id','duration','genre'])

    ## Combine file with metadata on tracks with the original songlist ID
    ## and drop some irrelevant columns
    df = pd.merge(df_meta,df_trackid,left_on='id',right_on='spotify_id',how='inner')

    columns_to_drop = ['track_href','search_artist','search_track','id','Unnamed: 0']
    df.drop(columns=columns_to_drop,inplace=True)
    
    ## Merge the music genre labels with the tracks including metadata. 
    df_out = pd.merge(df,df_label,left_on='track_id',right_on='track_id',how='inner')

    df_out.to_csv(result_file,index=False)

    return df_out


def create_train_test_validation():
    """This function will create a train, test and validation set """

    data_dir : Path = root / 'src' / 'data' / 'processed'
    input_file : str = 'labeled_data.csv'
    file_path : Path = data_dir / input_file

    if file_path.exists():
        logger.info(f"found file {input_file}, procceed with creating train, test and validation sets")
        try: 
            df = pd.read_csv(file_path)
        except:
            logger.info(f"an error occured while trying to pd.read_csv {file_path}")

    else:
        logger.info('Labeled data not found, creating a new file.')
        create_labeled_data()
    
    ## remove these columns
    columns_to_drop = [
        'type',
        'uri',
        'analysis_url',
        'time_signature',
        'track_id',
        'response_artist',
        'response_track',
        'spotify_id',
    ]

    df.drop(columns=columns_to_drop)

    ## make these columns categorical variables

    for col in ['key','mode','genre']:
        df[col] = df[col].astype('category')


    ##initialize encoders

    ohe = OneHotEncoder(sparse=False)
    scaler = StandardScaler()
    oe = OrdinalEncoder()

    ## Transform genre to ordinal coding
    ordinaL_columns = oe.fit_transform(df['genre'])

    ## apply OneHot encoding to categorical variables
    columns_to_onehot = ['key']
    onehot_columns = ohe.fit_transform(df[columns_to_onehot])

    ## Apply StandardScaler to numeric values
    columns_to_scale = [
        'danceabilitiy',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]
    
    scaled_columns = scaler.fit_transform(columns_to_scale)

    ## Not forget about the already binary feature 'mode'

    binary_columns = df['mode'].to_numpy()

    dataset = np.concatenate([onehot_columns,scaled_columns,binary_columns,ordinaL_columns],axis=1)


