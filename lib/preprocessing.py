import numpy as np
import pandas as pd
from astropy.table import Table

def drop_min_obs(train, min_obs=20):
    counts = train.groupby('object_id').size().rename('n_obs')
    keep_ids = counts[counts >= min_obs].index
    return train[train['object_id'].isin(keep_ids)].copy()

def find_peak_flux(train):
    idx = train.groupby('object_id')['flux'].idxmax()
    peaks_df = train.loc[idx, ['object_id', 'mjd', 'flux']].rename(columns={'mjd': 't_peak', 'flux': 'flux_peak'})
    return peaks_df.reset_index(drop=True)

def centralize_to_peak(train_df, peaks_df):
    shifted_df = train_df.merge(peaks_df[['object_id', 't_peak', 'flux_peak']], on='object_id', how='left')
    shifted_df['t_centered'] = shifted_df['mjd'] - shifted_df['t_peak']
    return shifted_df

def full_preprocess(args):
    metafilename = f'data/input/{args.meta_filename}.csv'
    metadata = Table.read(metafilename, format='csv')
    actualtrain_filename = f'data/input/{args.raw_filename}.csv'
    actualtrain = Table.read(actualtrain_filename, format='csv')

    print("Reading from:", metafilename, actualtrain_filename)

    train_df = actualtrain.to_pandas()

    print("Processing all data...")
    train_df_filtered = drop_min_obs(train_df, min_obs=70)
    print(f"After filtering: {train_df_filtered.shape}")

    peaks_df = find_peak_flux(train_df_filtered)
    print(f"Found peaks for {len(peaks_df)} objects")

    final_df = centralize_to_peak(train_df_filtered, peaks_df)
    print(f"Centralized light curves")

    metadata_df = metadata.to_pandas()
    final_df = final_df.merge(metadata_df, on='object_id', how='left')

    print(f"Final data shape: {final_df.shape}")
    print(f"Number of unique objects: {final_df['object_id'].nunique()}")
    print(f"Columns: {list(final_df.columns)}")

    output_path = f'data/output/{args.processed_filename}.csv'
    final_df.to_csv(output_path, index=False)
    print("Saved to:", output_path)