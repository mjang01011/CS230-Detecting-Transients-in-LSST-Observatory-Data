import numpy as np
import argparse

# these values are computed from the 80% training set, bc we don't want data leakage
FLUX_MEAN = 24.713942
FLUX_STD = 2882.842773

def normalize_npy(input_path, output_path, mean, std):
    """
    Apply global normalization to NPY file.

    Args:
        input_path: Path to input NPY file
        output_path: Path to save normalized NPY file
        mean: Global mean to subtract
        std: Global std to divide by
    """
    print("=" * 80)
    print("Applying Global Normalization to NPY File")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")
    print("=" * 80)

    data = np.load(input_path)
    print(f"Loaded data shape: {data.shape}")
    print("\nOriginal statistics (non-zero values):")
    non_zero_mask = data != 0
    non_zero_values = data[non_zero_mask]
    print(f"  Non-zero count: {non_zero_values.size:,}")
    print(f"  Mean: {non_zero_values.mean():.6f}")
    print(f"  Std: {non_zero_values.std():.6f}")
    print(f"  Min: {non_zero_values.min():.3f}")
    print(f"  Max: {non_zero_values.max():.3f}")

    data_normalized = data.copy()
    data_normalized[non_zero_mask] = (data[non_zero_mask] - mean) / std

    print("\nNormalized statistics (non-zero values):")
    non_zero_norm = data_normalized[non_zero_mask]
    print(f"  Mean: {non_zero_norm.mean():.6f}")
    print(f"  Std: {non_zero_norm.std():.6f}")
    print(f"  Min: {non_zero_norm.min():.3f}")
    print(f"  Max: {non_zero_norm.max():.3f}")

    np.save(output_path, data_normalized)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"  Shape: {data_normalized.shape}")
    print(f"  Normalized mean (non-zero): {non_zero_norm.mean():.6f}")
    print(f"  Normalized std (non-zero): {non_zero_norm.std():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply global normalization to NPY file')
    parser.add_argument('--input', type=str, default='data/output/test_data_ts2vec_train.npy',
                       help='Input NPY path')
    parser.add_argument('--output', type=str, default='data/output/test_data_ts2vec_train_global_norm.npy',
                       help='Output NPY path')
    parser.add_argument('--mean', type=float, default=FLUX_MEAN,
                       help=f'Global mean (default: {FLUX_MEAN})')
    parser.add_argument('--std', type=float, default=FLUX_STD,
                       help=f'Global std (default: {FLUX_STD})')
    args = parser.parse_args()

    normalize_npy(args.input, args.output, args.mean, args.std)
