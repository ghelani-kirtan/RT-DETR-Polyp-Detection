# -----------------------------------------------------------
# PREPARING THE CLASSIFICATION DATASET
# FROM THE DATASET REVIEWED BY THE CLIENT [client_data]
# -----------------------------------------------------------
import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm  # Progress bar

# Setup logging for readability and error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (configurable)
CLIENT_DATA_DIR = 'client_data'
POSITIVE_SUBDIR = 'positive_samples'
NEGATIVE_SUBDIR = 'negative_samples'
OUTPUT_DATASET_DIR = 'classification_dataset'  # Updated as requested
IMAGES_SUBDIR = 'images'
MASKS_SUBDIR = 'masks'

# Class-specific colors (RGB) - generic dict; add more classes as needed
CLASS_COLORS = {
    'adenoma': (255, 0, 0),         # Red
    'hyperplastic': (0, 255, 0),    # Green
    'benign': (157, 0, 255),        # Purple as requested
    'no pathology': (255, 255, 255) # White as requested
}

def convert_binary_mask_to_color(
    input_path: Path, 
    output_path: Path, 
    color: tuple
    ) -> bool:
    """
    Converts a binary mask (.tif or other) to a colored RGB mask (.jpg).
    
    Args:
        input_path (Path): Path to input binary mask.
        output_path (Path): Path to save colored mask.
        color (tuple): RGB color tuple (e.g., (255, 0, 0) for red).
    
    Returns:
        bool: True if successful, False if error.
    """
    try:
        mask = Image.open(input_path).convert('L')  # Convert to grayscale
        mask_array = np.array(mask)

        # Create RGB image with zeros (black background)
        colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

        # Apply color to foreground pixels (>0, assuming binary)
        foreground_pixels = mask_array > 0
        colored_mask[foreground_pixels] = color

        # Save as JPEG
        colored_image = Image.fromarray(colored_mask, 'RGB')
        colored_image.save(output_path, 'JPEG')
        return True
    except Exception as e:
        logger.error(f"Error converting mask {input_path}: {e}")
        return False

def convert_image_to_jpg(
    input_path: Path, 
    output_path: Path
    ) -> bool:
    """
    Converts an image to JPEG format if not already.
    
    Args:
        input_path (Path): Path to input image.
        output_path (Path): Path to save JPEG image.
    
    Returns:
        bool: True if successful or already JPEG, False if error.
    """
    try:
        if input_path.suffix.lower() in ('.jpg', '.jpeg'):
            shutil.copy(input_path, output_path)
        else:
            image = Image.open(input_path).convert('RGB')
            image.save(output_path, 'JPEG')
        return True
    except Exception as e:
        logger.error(f"Error converting image {input_path}: {e}")
        return False

def prepare_positive_samples(
    client_data_path: Path, 
    output_dataset_path: Path, 
    dry_run: bool = False
    ):

    positive_path = client_data_path / POSITIVE_SUBDIR
    output_images = output_dataset_path / IMAGES_SUBDIR
    output_masks = output_dataset_path / MASKS_SUBDIR
    
    if dry_run:
        logger.info("Dry-run mode: No files will be copied/converted.")
    else:
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
    
    if not positive_path.exists():
        logger.error(f"Positive samples directory not found: {positive_path}. Check your paths.")
        return
    
    copied_count = {'images': 0, 'masks': 0}
    
    for pathology in os.listdir(positive_path):
        pathology_lower = pathology.lower()
        if pathology_lower not in CLASS_COLORS:
            logger.warning(f"Skipping unknown class '{pathology}' (add to CLASS_COLORS if needed).")
            continue
        
        pathology_path = positive_path / pathology
        if not pathology_path.is_dir():
            continue
        
        images_src = pathology_path / 'images'
        masks_src = pathology_path / 'masks'
        
        if not images_src.exists() or not masks_src.exists():
            logger.warning(f"Missing images or masks for class '{pathology}' at {pathology_path}.")
            continue
        
        # Get list of images for tqdm progress
        img_files = [f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(img_files, desc=f"Processing class '{pathology}'"):
            basename = os.path.splitext(img_file)[0]
            mask_file_candidates = [f for f in os.listdir(masks_src) if os.path.splitext(f)[0] == basename]
            
            if not mask_file_candidates:
                logger.warning(f"No matching mask for image {img_file} in class '{pathology}'.")
                continue
            
            mask_file = mask_file_candidates[0]  # Take first match
            img_src_path = images_src / img_file
            mask_src_path = masks_src / mask_file
            
            img_dest_path = output_images / f"{basename}_{pathology_lower}.jpg"
            mask_dest_path = output_masks / f"{basename}_{pathology_lower}.jpg"
            color = CLASS_COLORS[pathology_lower]
            
            if dry_run:
                logger.info(f"Would process: Image {img_src_path} -> {img_dest_path}, Mask {mask_src_path} -> {mask_dest_path} (color: {color})")
            else:
                if convert_image_to_jpg(img_src_path, img_dest_path):
                    copied_count['images'] += 1
                if convert_binary_mask_to_color(mask_src_path, mask_dest_path, color):
                    copied_count['masks'] += 1
    
    logger.info(f"Aggregated {copied_count['images']} positive images and {copied_count['masks']} masks (dry_run={dry_run}).")

def copy_negative_samples(
    client_data_path: Path, 
    output_dataset_path: Path, 
    dry_run: bool = False
    ):
    
    negative_src = client_data_path / NEGATIVE_SUBDIR
    negative_dest = output_dataset_path / NEGATIVE_SUBDIR
    
    if not negative_src.exists():
        logger.warning(f"Negative samples directory not found: {negative_src}")
        return
    
    if dry_run:
        logger.info(f"Would copy negative samples from {negative_src} to {negative_dest}")
        return
    
    if negative_dest.exists():
        shutil.rmtree(negative_dest)  # Clear existing to avoid duplicates
    
    shutil.copytree(negative_src, negative_dest)
    copied_files = len(list(negative_dest.glob('*')))
    logger.info(f"Copied {copied_files} negative samples (dry_run={dry_run}).")

def main():
    parser = argparse.ArgumentParser(description="Prepare classification dataset with colored masks and negative samples.")
    parser.add_argument('--client_data_root', type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to parent of client_data directory (default: script's dir).")
    parser.add_argument('--output_root', type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to parent of output dataset directory (default: script's dir).")
    parser.add_argument('--dry_run', action='store_true', help="Preview actions without copying/converting files.")
    args = parser.parse_args()
    
    client_data_path = Path(args.client_data_root) / CLIENT_DATA_DIR
    output_dataset_path = Path(args.output_root) / OUTPUT_DATASET_DIR
    
    if not client_data_path.exists():
        logger.error(f"Client data directory not found at {client_data_path}. Use --client_data_root to set the correct path.")
        sys.exit(1)
    
    prepare_positive_samples(client_data_path, output_dataset_path, args.dry_run)
    copy_negative_samples(client_data_path, output_dataset_path, args.dry_run)
    
    logger.info("Dataset preparation completed.")

if __name__ == "__main__":
    main()