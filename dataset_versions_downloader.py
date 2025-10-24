# -----------------------------------------------------------
# SCRIPT FOR DOWNLOADING DATASET FROM API AND S3
# ORGANIZING INTO CLIENT_DATA FORMAT FOR FURTHER PROCESSING
# WITH MULTITHREADING FOR PARALLEL DOWNLOADS AND POST-DOWNLOAD VERIFICATION
# -----------------------------------------------------------
import os
import sys
import argparse
import logging
import requests
import json
from urllib.parse import urlparse, unquote
from pathlib import Path
import boto3
import botocore
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time  # For sleep in retries

# Setup logging for readability and error handling
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants (configurable)
DEFAULT_API_URL = "http://<server-ip4>:8000/api/v1/dataset-versions/list"  # Replace <server-ip4> with actual IP
CLIENT_DATA_DIR = "client_data"
POSITIVE_SUBDIR = "positive_samples"
NEGATIVE_SUBDIR = "negative_samples"
DATASET_VERSIONS_SUBDIR = "dataset_versions"
MULTIPART_THRESHOLD_MB = 25  # Threshold for multipart downloads
MULTIPART_CHUNKSIZE_MB = 25  # Chunk size for multipart
MAX_CONCURRENCY = 10  # Max threads for boto3 transfer per file
MAX_WORKERS = 20  # Reduced max threads to avoid connection pool exhaustion (adjust based on system)
USE_THREADS = True
MAX_POOL_CONNECTIONS = 200  # Further increased to handle more concurrent connections
RETRY_ATTEMPTS = 15  # Increased retry attempts for failed operations
RETRY_DELAY = 5  # Seconds to wait before retrying a failed download

# Class mappings (aligned with existing script)
CLASS_MAPPINGS = {
    "adenoma": "adenoma",
    "hyperplastic": "hyperplastic",
    "benign": "benign",
    "no pathology": "no_pathology",
    "multiple pathologies": "multiple_pathologies",
    # Add more as needed
}


def parse_s3_url(url: str) -> tuple:
    """
    Parses an S3 URL to extract bucket and key, decoding URL-encoded characters.

    Args:
        url (str): The S3 URL (presigned or standard).

    Returns:
        tuple: (bucket, key)
    """
    parsed = urlparse(url)
    if not parsed.netloc.endswith(".s3.amazonaws.com"):
        raise ValueError(f"Invalid S3 URL: {url}")
    bucket = parsed.netloc.split(".")[0]
    encoded_key = parsed.path.lstrip("/")
    key = unquote(
        encoded_key
    )  # Decode URL-encoded characters like %20 to space, %2C to comma
    return bucket, key


def download_file_from_s3(
    bucket: str,
    key: str,
    local_path: Path,
    s3_client,
    transfer_config: TransferConfig,
    dry_run: bool = False,
    max_retries: int = RETRY_ATTEMPTS,
) -> bool:
    """
    Downloads a file from S3 using boto3 with multipart support and retries.

    Args:
        bucket (str): S3 bucket name.
        key (str): S3 object key.
        local_path (Path): Local path to save the file.
        s3_client: boto3 S3 client.
        transfer_config (TransferConfig): Transfer configuration for multipart.
        dry_run (bool): If True, simulate download.
        max_retries (int): Max retry attempts.

    Returns:
        bool: True if successful, False if error after retries.
    """
    if dry_run:
        logger.info(f"Would download s3://{bucket}/{key} -> {local_path}")
        return True

    attempt = 0
    while attempt < max_retries:
        try:
            local_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            s3_client.download_file(
                Bucket=bucket, Key=key, Filename=str(local_path), Config=transfer_config
            )
            if local_path.exists() and local_path.stat().st_size > 0:
                return True
            else:
                logger.warning(
                    f"Downloaded file {local_path} is empty or missing after attempt {attempt + 1}"
                )
        except Exception as e:
            logger.error(
                f"Error downloading s3://{bucket}/{key} on attempt {attempt + 1}: {e}"
            )
        attempt += 1
        time.sleep(RETRY_DELAY * attempt)  # Exponential backoff
    return False


def fetch_dataset_versions(
    api_url: str, dataset_version_ids: list, page_index: int = 1, page_size: int = 0
) -> list:
    """
    Fetches dataset versions from the API.

    Args:
        api_url (str): API endpoint URL.
        dataset_version_ids (list): List of dataset version IDs.
        page_index (int): Page index for pagination.
        page_size (int): Page size (0 for all).

    Returns:
        list: List of dataset version data.
    """
    # body = {
    #     "page_index": page_index,
    #     "page_size": page_size,
    #     "order_by": "created_at",
    #     "order": "desc",
    #     "dataset_version_ids": dataset_version_ids,
    # }
    body = {"dataset_version_identifiers": dataset_version_ids}
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        resp_json = response.json()
        if not resp_json.get("success"):
            raise ValueError(f"API error: {resp_json.get('message')}")
        return resp_json.get("data", [])
    except Exception as e:
        logger.error(f"Error fetching from API {api_url}: {e}")
        sys.exit(1)


def prepare_download_task(
    frame: dict, video_name: str, output_client_path: Path, dry_run: bool = False
) -> list:
    """
    Prepares download tasks for a frame without downloading. Returns a list of download params.

    Args:
        frame (dict): Frame data from API.
        video_name (str): The video name to prefix filenames.
        output_client_path (Path): Path to client_data directory.
        dry_run (bool): If True, simulate actions.

    Returns:
        list: List of dicts with download params: {'bucket': str, 'key': str, 'local_path': Path, 'is_mask': bool, 'frame_id': int}
    """
    frame_id = frame.get("id")
    if frame.get("review_status") != "reviewed":
        logger.warning(
            f"Skipping unreviewed frame: {frame.get('image_id')} (ID: {frame_id})"
        )
        return []

    option = frame.get("option", "").lower()
    pathology_raw = (frame.get("pathology") or "").lower().replace(" ", "_")
    pathology = CLASS_MAPPINGS.get(pathology_raw, None)
    image_url = frame.get("image_url")
    mask_url = frame.get("mask_url")
    image_id = frame.get("image_id")

    if not image_url or not image_id:
        logger.warning(f"Missing image data for frame ID: {frame_id}")
        return []

    bucket, image_key = parse_s3_url(image_url)
    base_image_filename = os.path.basename(image_key)  # e.g., Hepatic, Sessile.png
    image_ext = os.path.splitext(base_image_filename)[1] or ".png"  # Ensure extension
    unique_image_filename = f"{video_name}_{base_image_filename}"

    download_tasks = []

    if option != "polyp" or pathology_raw == "no_pathology" or not mask_url:
        # Treat as negative sample (only image, no mask)
        neg_dir = output_client_path / NEGATIVE_SUBDIR
        if not dry_run:
            neg_dir.mkdir(parents=True, exist_ok=True)
        local_img_path = neg_dir / unique_image_filename
        download_tasks.append(
            {
                "bucket": bucket,
                "key": image_key,
                "local_path": local_img_path,
                "is_mask": False,
                "frame_id": frame_id,
                "category": "negative",
            }
        )

    else:
        # Positive sample
        if not pathology:
            logger.warning(
                f"Unknown pathology '{pathology_raw}' for frame {image_id} (ID: {frame_id}). Skipping."
            )
            return []

        pos_dir = output_client_path / POSITIVE_SUBDIR / pathology
        img_dir = pos_dir / "images"
        mask_dir = pos_dir / "masks"
        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

        local_img_path = img_dir / unique_image_filename
        download_tasks.append(
            {
                "bucket": bucket,
                "key": image_key,
                "local_path": local_img_path,
                "is_mask": False,
                "frame_id": frame_id,
                "category": pathology,
            }
        )

        _, mask_key = parse_s3_url(mask_url)
        base_mask_filename = os.path.basename(mask_key)  # e.g., Hepatic, Sessile.tif
        mask_ext = os.path.splitext(base_mask_filename)[1] or ".tif"  # Ensure extension
        unique_mask_filename = f"{video_name}_{base_mask_filename}"
        local_mask_path = mask_dir / unique_mask_filename
        download_tasks.append(
            {
                "bucket": bucket,
                "key": mask_key,
                "local_path": local_mask_path,
                "is_mask": True,
                "frame_id": frame_id,
                "category": pathology,
            }
        )

    return download_tasks


def verify_downloads(all_download_tasks, output_client_path: Path):
    """
    Verifies downloaded files, logs missing ones, and generates summary.

    Args:
        all_download_tasks (list): List of download task dicts.
        output_client_path (Path): Path to client_data directory.

    Returns:
        dict: Summary of counts and missing files.
    """
    missing_files = []
    category_counts = defaultdict(lambda: {"images": 0, "masks": 0})
    total_expected_images = 0
    total_expected_masks = 0

    for task in all_download_tasks:
        if task["is_mask"]:
            total_expected_masks += 1
        else:
            total_expected_images += 1

        if not task["local_path"].exists() or task["local_path"].stat().st_size == 0:
            missing_files.append(
                {
                    "frame_id": task["frame_id"],
                    "type": "mask" if task["is_mask"] else "image",
                    "path": task["local_path"],
                    "s3_key": task["key"],
                }
            )
        else:
            cat = task["category"]
            if task["is_mask"]:
                category_counts[cat]["masks"] += 1
            else:
                category_counts[cat]["images"] += 1

    summary = {
        "total_expected_images": total_expected_images,
        "total_expected_masks": total_expected_masks,
        "total_downloaded_images": sum(c["images"] for c in category_counts.values()),
        "total_downloaded_masks": sum(c["masks"] for c in category_counts.values()),
        "category_counts": dict(category_counts),
        "missing_files": missing_files,
    }

    logger.info("Download Verification Summary:")
    logger.info(
        f"Expected: {summary['total_expected_images']} images, {summary['total_expected_masks']} masks"
    )
    logger.info(
        f"Downloaded: {summary['total_downloaded_images']} images, {summary['total_downloaded_masks']} masks"
    )
    for cat, counts in summary["category_counts"].items():
        logger.info(
            f"Category '{cat}': {counts['images']} images, {counts['masks']} masks"
        )
    if missing_files:
        logger.warning(f"{len(missing_files)} files missing:")
        for mf in missing_files:
            logger.warning(
                f"Missing {mf['type']} for frame {mf['frame_id']}: {mf['path']} (S3: {mf['s3_key']})"
            )
    else:
        logger.info("No missing files.")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="""Download and organize dataset from API and S3 for classification preparation with multithreading and verification.
        Usage: 
            python dataset_versions_downloader.py --dataset_version_ids 4 --api_url http://<your-server-ip>:8000/api/v1/dataset-versions/list --output_root . --dry_run
        """
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=DEFAULT_API_URL,
        help="API URL for dataset versions (default: %(default)s). Replace <server-ip4> with actual IP.",
    )
    parser.add_argument(
        "--dataset_version_ids",
        type=int,
        nargs="+",
        required=True,
        help="List of dataset version IDs to download (space-separated).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        # default=os.path.dirname(os.path.abspath(__file__)),
        default=os.getcwd(),
        help="Path to output root directory (default: script's dir). The final path will be <output_root>/dataset_versions/<version_title>/client_data.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview actions without downloading files.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of threads for parallel downloads (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry-run mode: No files will be downloaded.")

    # Initialize boto3 client with increased pool connections and retries
    client_config = botocore.config.Config(
        max_pool_connections=MAX_POOL_CONNECTIONS,
        retries={"max_attempts": RETRY_ATTEMPTS, "mode": "adaptive"},
    )
    s3_client = boto3.client("s3", config=client_config)

    # Transfer config for multipart
    transfer_config = TransferConfig(
        multipart_threshold=1024 * 1024 * MULTIPART_THRESHOLD_MB,
        max_concurrency=MAX_CONCURRENCY,
        multipart_chunksize=1024 * 1024 * MULTIPART_CHUNKSIZE_MB,
        use_threads=USE_THREADS,
    )

    # Fetch dataset versions
    dataset_versions = fetch_dataset_versions(args.api_url, args.dataset_version_ids)
    if not dataset_versions:
        logger.info("No dataset versions found.")
        return

    for version in tqdm(dataset_versions, desc="Processing dataset versions"):
        version_id = version.get("id")
        version_title = version.get(
            "dataset_version_title", f"version_{version_id}"
        ).replace(
            " ", "_"
        )  # Fallback and sanitize for filesystem
        logger.info(
            f"Processing dataset version ID: {version_id} (Title: {version_title})"
        )

        # Set version-specific output path: output_root/dataset_versions/version_title/client_data
        output_version_path = (
            Path(args.output_root) / DATASET_VERSIONS_SUBDIR / version_title
        )
        output_client_path = output_version_path / CLIENT_DATA_DIR

        # Collect all download tasks
        all_download_tasks = []
        video_folders = version.get("video_folders", [])
        for video in video_folders:
            video_name = video.get(
                "video_name", f"video_{video.get('video_folder_id')}"
            )  # Fallback if no name
            frames = video.get("frames", [])
            for frame in frames:
                tasks = prepare_download_task(
                    frame, video_name, output_client_path, args.dry_run
                )
                all_download_tasks.extend(tasks)

        successful_counts = {"images": 0, "masks": 0}

        # Download in parallel using ThreadPoolExecutor
        if all_download_tasks:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        download_file_from_s3,
                        task["bucket"],
                        task["key"],
                        task["local_path"],
                        s3_client,
                        transfer_config,
                        args.dry_run,
                    ): task
                    for task in all_download_tasks
                }
                for future in tqdm(
                    as_completed(future_to_task),
                    total=len(all_download_tasks),
                    desc="Downloading files",
                ):
                    task = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            if task["is_mask"]:
                                successful_counts["masks"] += 1
                            else:
                                successful_counts["images"] += 1
                    except Exception as e:
                        logger.error(
                            f"Exception in download task for frame {task['frame_id']}: {e}"
                        )

        logger.info(
            f"Initially successful: {successful_counts['images']} images and {successful_counts['masks']} masks (dry_run={args.dry_run})."
        )

        # Verify and log summary
        summary = verify_downloads(all_download_tasks, output_client_path)

        # Optionally retry missing
        if summary["missing_files"] and not args.dry_run:
            logger.info(f"Retrying {len(summary['missing_files'])} missing files...")
            retry_tasks = []
            for mf in summary["missing_files"]:
                task = next(
                    (
                        t
                        for t in all_download_tasks
                        if str(t["local_path"]) == str(mf["path"])
                    ),
                    None,
                )
                if task:
                    retry_tasks.append(task)

            if retry_tasks:
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    future_to_task = {
                        executor.submit(
                            download_file_from_s3,
                            task["bucket"],
                            task["key"],
                            task["local_path"],
                            s3_client,
                            transfer_config,
                            args.dry_run,
                        ): task
                        for task in retry_tasks
                    }
                    for future in tqdm(
                        as_completed(future_to_task),
                        total=len(retry_tasks),
                        desc="Retrying missing files",
                    ):
                        task = future_to_task[future]
                        try:
                            success = future.result()
                            if success:
                                if task["is_mask"]:
                                    successful_counts["masks"] += 1
                                else:
                                    successful_counts["images"] += 1
                        except Exception as e:
                            logger.error(
                                f"Exception in retry for frame {task['frame_id']}: {e}"
                            )

            # Re-verify after retry
            verify_downloads(all_download_tasks, output_client_path)

    logger.info("Download, organization, and verification completed.")


if __name__ == "__main__":
    main()
