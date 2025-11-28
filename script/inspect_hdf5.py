#!/usr/bin/env python3
"""
Script to inspect and print the structure of HDF5 files.

Usage:
    python scripts/inspect_hdf5.py <hdf5_file_path>
    python scripts/inspect_hdf5.py <hdf5_file_path> --show-data   # also print sample data values
    python scripts/inspect_hdf5.py <hdf5_file_path> --max-depth 3 # limit recursion depth

Examples:
    python scripts/inspect_hdf5.py data/episode0.hdf5
    python scripts/inspect_hdf5.py RoboTwin/data/beat_block_hammer/demo_randomized/data/episode0.hdf5 --show-data
"""

import argparse
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    print("Error: h5py is not installed. Install it with: pip install h5py")
    sys.exit(1)

import numpy as np


def format_shape(shape):
    """Format shape tuple for display."""
    return f"({', '.join(str(d) for d in shape)})" if shape else "()"


def format_dtype(dtype):
    """Format dtype for display."""
    return str(dtype)


def format_size(nbytes):
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def print_attrs(obj, indent=""):
    """Print attributes of an HDF5 object."""
    if len(obj.attrs) > 0:
        print(f"{indent}  [Attributes]")
        for key, val in obj.attrs.items():
            val_str = str(val)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"{indent}    {key}: {val_str}")


def print_dataset_info(name, dataset, indent="", show_data=False):
    """Print information about a dataset."""
    shape_str = format_shape(dataset.shape)
    dtype_str = format_dtype(dataset.dtype)
    size_str = format_size(dataset.nbytes) if hasattr(dataset, 'nbytes') else "N/A"
    
    # Compression info
    compression = dataset.compression if dataset.compression else "None"
    chunks = dataset.chunks if dataset.chunks else "None"
    
    print(f"{indent}📊 {name}")
    print(f"{indent}   Shape: {shape_str}  |  Dtype: {dtype_str}  |  Size: {size_str}")
    print(f"{indent}   Compression: {compression}  |  Chunks: {chunks}")
    
    print_attrs(dataset, indent)
    
    if show_data:
        try:
            data = dataset[...]
            if data.size <= 10:
                print(f"{indent}   Data: {data}")
            else:
                # Show first few and last few elements
                flat = data.flatten()
                print(f"{indent}   Data (first 5): {flat[:5]}")
                print(f"{indent}   Data (last 5):  {flat[-5:]}")
                print(f"{indent}   Min: {np.min(data):.6g}  |  Max: {np.max(data):.6g}  |  Mean: {np.mean(data):.6g}")
        except Exception as e:
            print(f"{indent}   Data: <error reading: {e}>")


def print_group_info(name, group, indent=""):
    """Print information about a group."""
    n_items = len(group)
    print(f"{indent}📁 {name}/ ({n_items} items)")
    print_attrs(group, indent)


def traverse_hdf5(obj, path="/", indent="", depth=0, max_depth=None, show_data=False):
    """Recursively traverse and print HDF5 structure."""
    if max_depth is not None and depth > max_depth:
        print(f"{indent}  ... (max depth reached)")
        return
    
    for key in obj.keys():
        item = obj[key]
        item_path = f"{path}{key}" if path == "/" else f"{path}/{key}"
        
        if isinstance(item, h5py.Group):
            print_group_info(key, item, indent)
            traverse_hdf5(item, item_path, indent + "  ", depth + 1, max_depth, show_data)
        elif isinstance(item, h5py.Dataset):
            print_dataset_info(key, item, indent, show_data)
        else:
            print(f"{indent}❓ {key}: <unknown type: {type(item)}>")


def inspect_hdf5(file_path, max_depth=None, show_data=False):
    """Main function to inspect an HDF5 file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    if not file_path.suffix.lower() in ['.hdf5', '.h5', '.hdf']:
        print(f"Warning: File extension '{file_path.suffix}' is not a typical HDF5 extension")
    
    print("=" * 70)
    print(f"HDF5 File: {file_path}")
    print(f"File Size: {format_size(file_path.stat().st_size)}")
    print("=" * 70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n📦 Root /")
            print_attrs(f, "")
            print()
            traverse_hdf5(f, "/", "", 0, max_depth, show_data)
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Inspection complete.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and print the structure of HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the HDF5 file to inspect"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=None,
        help="Maximum depth to traverse (default: unlimited)"
    )
    parser.add_argument(
        "--show-data", "-s",
        action="store_true",
        help="Show sample data values for each dataset"
    )
    
    args = parser.parse_args()
    
    success = inspect_hdf5(args.file, args.max_depth, args.show_data)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
