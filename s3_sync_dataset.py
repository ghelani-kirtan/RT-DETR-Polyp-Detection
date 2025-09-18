import os
import argparse
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import concurrent.futures
from functools import partial
from boto3.s3.transfer import TransferConfig

# Load environment variables from .env file
load_dotenv()

# AWS credentials from .env (optional; prefer IAM role)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

# Local Details:
DEFAULT_LOCAL = 'dataset'

# S3 details
BUCKET_NAME = 'seekiq-s3-dev'
DEFAULT_PREFIX = 'polyp_data_ml/dataset_classification/'


# Transfer configuration for multipart uploads/downloads
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,  # 8MB threshold for multipart
    max_concurrency=10,  # Number of threads for multipart transfers
    multipart_chunksize=8 * 1024 * 1024,  # Chunk size for multipart
    num_download_attempts=10,
    max_io_queue=100,
    io_chunksize=256 * 1024,
    use_threads=True
)

def create_s3_client():
    """Create and return a boto3 S3 client."""
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        return session.client('s3')
    else:
        # Fallback to default (IAM role)
        return boto3.client('s3')

def dir_exists_in_s3(s3_client, bucket, prefix):
    """Check if a 'directory' (prefix) exists in S3 by seeing if it has any objects."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response

def get_s3_subdirs(s3_client, bucket, prefix):
    """Get list of first-level subdirectories under the S3 prefix."""
    subdirs = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        for common_prefix in page.get('CommonPrefixes', []):
            subdir = common_prefix['Prefix'].rstrip('/').split('/')[-1]
            subdirs.add(subdir)
    return list(subdirs)

def upload_single(s3_client, bucket, local_file, s3_key):
    """Upload a single file to S3 with error handling."""
    try:
        s3_client.upload_file(local_file, bucket, s3_key, Config=TRANSFER_CONFIG)
        print(f"Uploaded {local_file} to {s3_key}")
    except ClientError as e:
        print(f"Error uploading {local_file}: {e}")

def upload_command(args):
    """Handle upload: Sync local dataset subdirs to S3, skipping duplicate subdirs."""
    s3_client = create_s3_client()
    local_dir = args.local_dir
    s3_prefix = args.s3_prefix.rstrip('/') + '/'

    if not os.path.isdir(local_dir):
        print(f"Local directory {local_dir} does not exist.")
        return

    # Get local subdirs (e.g., images, masks, negative_samples)
    local_subdirs = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]

    for subdir in local_subdirs:
        local_subdir_path = os.path.join(local_dir, subdir)
        s3_sub_prefix = s3_prefix + subdir + '/'

        if dir_exists_in_s3(s3_client, BUCKET_NAME, s3_sub_prefix):
            print(f"Skipping upload for {subdir} as it already exists in S3.")
            continue

        print(f"Uploading {subdir} to S3...")

        # Collect all files to upload
        file_list = []
        for root, _, files in os.walk(local_subdir_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_dir)
                # Normalize to forward slashes for S3
                relative_path = relative_path.replace('\\', '/')
                s3_key = s3_prefix + relative_path
                file_list.append((local_file, s3_key))

        # Parallel upload
        upload_func = partial(upload_single, s3_client, BUCKET_NAME)
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(lambda args: upload_func(*args), file_list))

def download_single(s3_client, bucket, key, local_path):
    """Download a single file from S3 with error handling."""
    try:
        s3_client.download_file(bucket, key, local_path, Config=TRANSFER_CONFIG)
        print(f"Downloaded {key} to {local_path}")
    except ClientError as e:
        print(f"Error downloading {key}: {e}")

def download_command(args):
    """Handle download: Sync S3 dataset subdirs to local, skipping duplicate subdirs."""
    s3_client = create_s3_client()
    local_dir = args.local_dir
    s3_prefix = args.s3_prefix.rstrip('/') + '/'

    os.makedirs(local_dir, exist_ok=True)

    # Get S3 subdirs (e.g., images, masks, negative_samples)
    s3_subdirs = get_s3_subdirs(s3_client, BUCKET_NAME, s3_prefix)

    for subdir in s3_subdirs:
        local_subdir_path = os.path.join(local_dir, subdir)
        if os.path.isdir(local_subdir_path):
            print(f"Skipping download for {subdir} as it already exists locally.")
            continue

        print(f"Downloading {subdir} from S3...")

        # Collect all objects to download
        object_list = []
        s3_sub_prefix = s3_prefix + subdir + '/'
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_sub_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                relative_key = key[len(s3_prefix):]
                # Split by '/' and use os.path.join for local OS compatibility
                parts = relative_key.split('/')
                local_path = os.path.join(local_dir, *parts)
                object_list.append((key, local_path))

        # Create all necessary directories first
        for _, local_path in object_list:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Parallel download
        download_func = partial(download_single, s3_client, BUCKET_NAME)
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(lambda args: download_func(*args), object_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync dataset with S3.",
        epilog="""
Examples:
  python s3_sync_data.py upload --local_dir my_dataset --s3-prefix polyp_data_ml/data_v2/
    (Uploads from 'my_dataset' subdirs to S3 prefix 'polyp_data_ml/data_v2/', skipping duplicates)

  python s3_sync_data.py download --local_dir dataset --s3-prefix polyp_data_ml/dataset_classification/
    (Downloads from S3 prefix 'polyp_data_ml/dataset_classification/' subdirs to 'dataset', skipping duplicates)

Note: The script uses the bucket 'seekiq-s3-dev'. Prefixes are created automatically on upload if they don't exist.
        """
    )
    parser.add_argument('--local_dir', type=str, default=DEFAULT_LOCAL,
                        help="Path to local dataset directory (default: 'dataset')")
    parser.add_argument('--s3-prefix', type=str, default=DEFAULT_PREFIX,
                        help="S3 prefix (default: 'polyp_data_ml/dataset_classification/')")

    subparsers = parser.add_subparsers(dest='command', required=True)

    upload_parser = subparsers.add_parser('upload', help="Upload to S3, skipping duplicates")
    download_parser = subparsers.add_parser('download', help="Download from S3, skipping duplicates")

    args = parser.parse_args()

    if args.command == 'upload':
        upload_command(args)
    elif args.command == 'download':
        download_command(args)