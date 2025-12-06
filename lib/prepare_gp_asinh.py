import os
import numpy as np
import pandas as pd
import argparse
from astropy.table import Table
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def drop_min_obs(train, min_obs=20):
    """Filter objects with minimum observations"""
    counts = train.groupby('object_id').size().rename('n_obs')
    keep_ids = counts[counts >= min_obs].index
    return train[train['object_id'].isin(keep_ids)].copy()


def find_peak_flux(train):
    """Find peak flux time for each object"""
    idx = train.groupby('object_id')['flux'].idxmax()
    peaks_df = train.loc[idx, ['object_id', 'mjd', 'flux']].rename(
        columns={'mjd': 't_peak', 'flux': 'flux_peak'}
    )
    return peaks_df.reset_index(drop=True)


def centralize_to_peak(train_df, peaks_df):
    """Center time to peak flux"""
    shifted_df = train_df.merge(
        peaks_df[['object_id', 't_peak', 'flux_peak']],
        on='object_id',
        how='left'
    )
    shifted_df['t_centered'] = shifted_df['mjd'] - shifted_df['t_peak']
    return shifted_df


def asinh_normalize_flux(flux, scale_factor=10.0):
    """Apply asinh normalization to handle negative flux values"""
    return np.arcsinh(flux / scale_factor)


def gp_interpolate_passband(times, flux, flux_err, grid_times):
    """
    Use Gaussian Process to interpolate flux onto a regular time grid
    """
    if len(times) == 0:
        return np.zeros(len(grid_times))
    if len(times) == 1:
        return np.full(len(grid_times), flux[0])
    kernel = Matern(length_scale=10.0, nu=1.5)
    alpha = np.median(flux_err) if len(flux_err) > 0 else 1.0
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=0,
        normalize_y=False
    )
    gp.fit(times.reshape(-1, 1), flux)
    interpolated_flux = gp.predict(grid_times.reshape(-1, 1), return_std=False)
    return interpolated_flux


def convert_to_ts2vec_format_gp_asinh(df, max_length=200, scale_factor=10.0, time_step=1.0):
    """
    Convert light curve data to TS2Vec format using GP interpolation and asinh normalization

    Args:
        df: DataFrame with light curve observations
        max_length: Maximum sequence length (in days)
        scale_factor: Asinh normalization scale factor
        time_step: Time step for regular grid (days)

    Returns:
        data: Array of shape (n_objects, max_length, n_passbands)
        valid_ids: List of object IDs
    """
    object_ids = df['object_id'].unique()
    passbands = sorted(df['passband'].unique())
    n_passbands = len(passbands)

    data = np.zeros((len(object_ids), max_length, n_passbands))
    valid_ids = []

    print(f"Converting {len(object_ids)} objects to TS2Vec format")
    print(f"Passbands: {passbands}")
    print(f"Features: {n_passbands}")

    for idx, obj_id in enumerate(tqdm(object_ids)):
        obj_data = df[df['object_id'] == obj_id]

        # Define time grid centered on peak
        t_min = max(obj_data['t_centered'].min(), -max_length/2 * time_step)
        t_max = min(obj_data['t_centered'].max(), max_length/2 * time_step)

        grid_times = np.arange(t_min, t_max + time_step, time_step)

        # Make sure grid is exactly max_length
        if len(grid_times) > max_length:
            start_idx = len(grid_times) // 2 - max_length // 2
            grid_times = grid_times[start_idx:start_idx + max_length]
        elif len(grid_times) < max_length:
            # Pad symmetrically around peak
            pad_before = (max_length - len(grid_times)) // 2
            pad_after = max_length - len(grid_times) - pad_before
            grid_times = np.concatenate([
                np.arange(grid_times[0] - pad_before * time_step, grid_times[0], time_step),
                grid_times,
                np.arange(grid_times[-1] + time_step, grid_times[-1] + (pad_after + 1) * time_step, time_step)
            ])[:max_length]

        # Process each passband separately
        for pb_idx, passband in enumerate(passbands):
            pb_data = obj_data[obj_data['passband'] == passband].sort_values('t_centered')

            if len(pb_data) > 0:
                times = pb_data['t_centered'].values
                flux = pb_data['flux'].values
                flux_err = pb_data['flux_err'].values

                # Apply asinh normalization first
                flux_norm = asinh_normalize_flux(flux, scale_factor)

                # Then interpolate with GP
                interpolated_flux = gp_interpolate_passband(times, flux_norm, flux_err, grid_times)

                data[idx, :, pb_idx] = interpolated_flux

        valid_ids.append(obj_id)

    return data, valid_ids


def process_batch(batch_file, max_length=200, min_obs=70, scale_factor=None,
                  time_step=1.0, sub_batches=5, output_dir='data/output'):
    """
    Process a single batch file with GP + Asinh preprocessing
    Split into sub-batches for memory efficiency
    """
    # same preprocessing as our defeault method
    print(f"\nProcessing {batch_file}")
    raw_data = Table.read(batch_file, format='csv')
    df = raw_data.to_pandas()
    print(f"  Loaded {len(df)} observations for {df['object_id'].nunique()} objects")
    df = drop_min_obs(df, min_obs=min_obs)
    print(f"  After filtering: {df['object_id'].nunique()} objects")
    peaks_df = find_peak_flux(df)
    df = centralize_to_peak(df, peaks_df)

    # memory can explode if you dont split
    object_ids_all = df['object_id'].unique()
    n_objects = len(object_ids_all)
    sub_batch_size = n_objects // sub_batches

    batch_num = os.path.basename(batch_file).replace('test_set_batch', '').replace('.csv', '')
    temp_dir = f'{output_dir}/temp_gp_asinh_batch{batch_num}'
    os.makedirs(temp_dir, exist_ok=True)

    all_data = []
    all_ids = []

    for sub_idx in range(sub_batches):
        start_idx = sub_idx * sub_batch_size
        end_idx = n_objects if sub_idx == sub_batches - 1 else (sub_idx + 1) * sub_batch_size
        sub_object_ids = object_ids_all[start_idx:end_idx]
        sub_file = f'{temp_dir}/sub_batch{sub_idx}.npy'
        sub_ids_file = f'{temp_dir}/sub_batch{sub_idx}_ids.npy'

        # skip already completed ones
        if os.path.exists(sub_file) and os.path.exists(sub_ids_file):
            print(f"Sub-batch {sub_idx+1}/{sub_batches} already exists, loading")
            sub_data = np.load(sub_file)
            sub_ids = np.load(sub_ids_file)
        else:
            print(f"Processing sub-batch {sub_idx+1}/{sub_batches} ({len(sub_object_ids)} objects)")
            df_sub = df[df['object_id'].isin(sub_object_ids)]

            sub_data, sub_ids = convert_to_ts2vec_format_gp_asinh(
                df_sub,
                max_length=max_length,
                scale_factor=scale_factor,
                time_step=time_step
            )
            np.save(sub_file, sub_data)
            np.save(sub_ids_file, np.array(sub_ids))
            print(f"Saved sub-batch {sub_idx+1} to {sub_file}")

        all_data.append(sub_data)
        all_ids.extend(sub_ids)

    # combine subbatches into single file
    combined_data = np.concatenate(all_data, axis=0)

    return combined_data, all_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GP + Asinh preprocessing for PLAsTiCC'
    )
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing test_set_batch*.csv files')
    parser.add_argument('--output_dir', type=str, default='data/output',
                       help='Output directory for processed files')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximum sequence length in days')
    parser.add_argument('--min_obs', type=int, default=70,
                       help='Minimum observations to keep object')
    parser.add_argument('--scale_factor', type=float, default=None,
                       help='Asinh normalization scale factor (auto-calculated if not provided)')
    parser.add_argument('--time_step', type=float, default=1.0,
                       help='Time step for regular grid (days)')
    parser.add_argument('--sub_batches', type=int, default=5,
                       help='Number of sub-batches per batch for crash resilience')
    parser.add_argument('--process_batches', type=str, default='1-11',
                       help='Batches to process (e.g., "1-10" or "11")')

    args = parser.parse_args()

    if '-' in args.process_batches:
        start, end = map(int, args.process_batches.split('-'))
        batches = list(range(start, end + 1))
    else:
        batches = [int(args.process_batches)]

    print("GP + Asinh preprocessing")
    print(f"Config:")
    print(f"Max length: {args.max_length}")
    print(f"Time step: {args.time_step}")
    print(f"Min observations: {args.min_obs}")
    print(f"Asinh scale factor: {args.scale_factor if args.scale_factor else 'auto (median flux_err)'}")
    print(f"Interpolation: Gaussian Process (sklearn)")
    print(f"Sub-batches per batch: {args.sub_batches}")
    print(f"Batches: {batches}")
    if len(batches) > 1:
        print(f"\nProcessing {len(batches)} batches")
        temp_dir = f'{args.output_dir}/temp_batches_gp_asinh'
        os.makedirs(temp_dir, exist_ok=True)

        all_ids = []

        for batch_num in batches:
            batch_file = f'{args.data_dir}/test_set_batch{batch_num}.csv'

            if not os.path.exists(batch_file):
                print(f"Warning: {batch_file} not found, skipping")
                continue

            data, ids = process_batch(
                batch_file,
                max_length=args.max_length,
                min_obs=args.min_obs,
                scale_factor=args.scale_factor,
                time_step=args.time_step,
                sub_batches=args.sub_batches,
                output_dir=args.output_dir
            )
            temp_path = f'{temp_dir}/batch{batch_num}.npy'
            np.save(temp_path, data)
            all_ids.extend(ids)
            print(f"  Saved batch {batch_num} to temp file")
            del data # need to save memory..

        all_batches = []
        for batch_num in batches:
            temp_path = f'{temp_dir}/batch{batch_num}.npy'
            if os.path.exists(temp_path):
                batch_data = np.load(temp_path)
                all_batches.append(batch_data)

        combined_data = np.concatenate(all_batches, axis=0)
        all_ids = np.array(all_ids)

        # save final result
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = f'{args.output_dir}/test_data_ts2vec_gp_asinh.npy'
        ids_path = f'{args.output_dir}/test_data_ts2vec_gp_asinh_ids.npy'

        np.save(output_path, combined_data)
        np.save(ids_path, all_ids)
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Total objects: {len(all_ids)}")
        print(f"Saved to {output_path}")


    # for single batch (not recommended due to memory issues (exploded w/ 32gb ram))
    else:
        batch_file = f'{args.data_dir}/test_set_batch{batches[0]}.csv'
        data, ids = process_batch(
            batch_file,
            max_length=args.max_length,
            min_obs=args.min_obs,
            scale_factor=args.scale_factor,
            time_step=args.time_step,
            sub_batches=args.sub_batches,
            output_dir=args.output_dir
        )
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = f'{args.output_dir}/test_data_batch{batches[0]}_gp_asinh.npy'
        ids_path = f'{args.output_dir}/test_data_batch{batches[0]}_gp_asinh_ids.npy'

        np.save(output_path, data)
        np.save(ids_path, np.array(ids))
        print(f"Data shape: {data.shape}")
        print(f"Total objects: {len(ids)}")
        print(f"Saved to {output_path}")

