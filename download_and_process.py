import os
import pandas as pd
import numpy as np

def download_plasticc_data():
    """
    Download PLAsTiCC dataset using Kaggle API
    Make sure you have kaggle.json in ~/.kaggle/ or C:/Users/<username>/.kaggle/
    """
    print("Downloading PLAsTiCC-2018 dataset...")
    os.system("kaggle competitions download -c PLAsTiCC-2018")
    print("Extracting files...")
    os.system("unzip -q PLAsTiCC-2018.zip -d data/raw/")
    print("Download complete!")

def drop_min_obs(train, min_obs=20):
    counts = train.groupby('object_id').size().rename('n_obs')
    keep_ids = counts[counts >= min_obs].index
    return train[train['object_id'].isin(keep_ids)].copy()

def find_peak_flux(train):
    idx = train.groupby('object_id')['flux'].idxmax()
    peaks_df = train.loc[idx, ['object_id', 'mjd', 'flux']].rename(
        columns={'mjd': 't_peak', 'flux': 'flux_peak'}
    )
    return peaks_df.reset_index(drop=True)

def centralize_to_peak(train_df, peaks_df):
    shifted_df = train_df.merge(
        peaks_df[['object_id', 't_peak', 'flux_peak']],
        on='object_id',
        how='left'
    )
    shifted_df['t_centered'] = shifted_df['mjd'] - shifted_df['t_peak']
    return shifted_df

def process_data(input_path='data/raw/training_set.csv',
                 metadata_path='data/raw/training_set_metadata.csv',
                 output_path='data/processed/processed_training.csv',
                 min_obs=70):

    print("Loading data...")
    train_df = pd.read_csv(input_path)
    metadata = pd.read_csv(metadata_path)

    print(f"Initial data shape: {train_df.shape}")

    print(f"Filtering objects with fewer than {min_obs} observations...")
    train_df = drop_min_obs(train_df, min_obs=min_obs)
    print(f"After filtering: {train_df.shape}")

    print("Finding peak flux for each object...")
    peaks_df = find_peak_flux(train_df)

    print("Centralizing light curves to peak...")
    processed_df = centralize_to_peak(train_df, peaks_df)

    print("Merging with metadata...")
    processed_df = processed_df.merge(metadata, on='object_id', how='left')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving processed data to {output_path}...")
    processed_df.to_csv(output_path, index=False)

    print(f"Final processed data shape: {processed_df.shape}")
    print(f"Number of unique objects: {processed_df['object_id'].nunique()}")
    print("\nProcessing complete!")

    return processed_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download dataset first')
    parser.add_argument('--input', default='data/raw/training_set.csv', help='Input CSV path')
    parser.add_argument('--metadata', default='data/raw/training_set_metadata.csv', help='Metadata CSV path')
    parser.add_argument('--output', default='data/processed/processed_training.csv', help='Output CSV path')
    parser.add_argument('--min_obs', type=int, default=70, help='Minimum observations per object')

    args = parser.parse_args()

    if args.download:
        download_plasticc_data()

    process_data(args.input, args.metadata, args.output, args.min_obs)
