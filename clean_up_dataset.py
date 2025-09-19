import os
import sys
import argparse
from tqdm import tqdm

# Constants - Customize these as needed
DATASET_DIR = "dataset"  # Root directory of the dataset
IMAGES_SUBDIR = "images"  # Subdirectory for images
MASKS_SUBDIR = "masks"  # Subdirectory for masks


def get_file_stems(directory):
    """
    Retrieve a set of file stems (filenames without extensions) from the given directory.

    Args:
        directory (str): Path to the directory to scan.

    Returns:
        set: Set of file stems.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    stems = set()
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            stem = os.path.splitext(filename)[0]
            stems.add(stem)
    return stems


def remove_unmatched_masks(images_dir, masks_dir):
    """
    Remove mask files that do not have a corresponding image file with the same stem.

    Args:
        images_dir (str): Path to the images directory.
        masks_dir (str): Path to the masks directory.
    """
    image_stems = get_file_stems(images_dir)

    removed_count = 0
    mask_files = [
        f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))
    ]

    for filename in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(masks_dir, filename)
        stem = os.path.splitext(filename)[0]
        if stem not in image_stems:
            os.remove(mask_path)
            print(f"Removed unmatched mask: {filename}")
            removed_count += 1

    print(f"Total masks removed: {removed_count}")
    if removed_count == 0:
        print("No unmatched masks found.")


def count_polyp_types(directory):
    """
    Count the number of files grouped by polyp types based on file stem endings.

    Args:
        directory (str): Path to the directory to scan.

    Returns:
        dict: Dictionary with counts for 'adenoma', 'hyperplastic', and 'other'.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    types = {"adenoma": 0, "hyperplastic": 0, "other": 0}
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            stem = os.path.splitext(filename)[0]
            if stem.endswith("_adenoma"):
                types["adenoma"] += 1
            elif stem.endswith("_hyperplastic"):
                types["hyperplastic"] += 1
            else:
                types["other"] += 1
    return types


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Clean up dataset by removing unmatched masks and count polyp types."
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count the polyp types without removing unmatched masks.",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Also print the overall count in addition to group-wise counts.",
    )
    args = parser.parse_args()

    images_dir = os.path.join(DATASET_DIR, IMAGES_SUBDIR)
    masks_dir = os.path.join(DATASET_DIR, MASKS_SUBDIR)

    try:
        if not args.count_only:
            remove_unmatched_masks(images_dir, masks_dir)

        # Count polyp types from images directory
        types = count_polyp_types(images_dir)

        print("Group-wise counts:")
        print(f"Adenoma: {types['adenoma']}")
        print(f"Hyperplastic: {types['hyperplastic']}")
        print(f"Other: {types['other']}")

        if args.overall:
            overall = sum(types.values())
            print(f"Overall count: {overall}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
