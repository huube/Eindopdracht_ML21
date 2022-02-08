from pathlib import Path
from loguru import logger
import pandas as pd

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