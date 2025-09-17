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
AWS_REGION = os.getenv('AWS_REGION')

# S3 details
BUCKET_NAME = 'seekiq-s3-dev'
DEFAULT_PREFIX = 'polyp_data_ml/models/detection/custom_rtdetr_r18vd_polyp/'


# Transfer configuration for multipart uploads/downloads
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,  # 5MB threshold for multipart (suitable for small to large files)
    max_concurrency=10,  # Number of threads for multipart transfers
    multipart_chunksize=5 * 1024 * 1024,  # 5MB chunk size for multipart (optimized for 150-300MB files)
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
            region_name=AWS_REGION
        )
        return session.client('s3')
    else:
        # Fallback to default (IAM role)
        return boto3.client('s3')

def upload_single(s3_client, bucket, local_file, s3_key):
    """Upload a single file to S3 with error handling."""
    try:
        s3_client.upload_file(local_file, bucket, s3_key, Config=TRANSFER_CONFIG)
        print(f"Uploaded {local_file} to s3://{bucket}/{s3_key}")
    except ClientError as e:
        print(f"Error uploading {local_file}: {e}")

def upload_directory(args):
    """Upload the local directory to S3 with parallel processing."""
    s3_client = create_s3_client()
    local_dir = args.local_dir
    s3_prefix = args.s3_prefix.rstrip('/') + '/'

    if not os.path.isdir(local_dir):
        print(f"Local directory {local_dir} does not exist.")
        return

    # Collect all files to upload
    file_list = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_dir)
            # Normalize to forward slashes for S3
            relative_path = relative_path.replace('\\', '/')
            s3_key = s3_prefix + relative_path
            file_list.append((local_file, s3_key))

    print(f"Uploading {len(file_list)} files to s3://{BUCKET_NAME}/{s3_prefix}...")

    # Parallel upload
    upload_func = partial(upload_single, s3_client, BUCKET_NAME)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(lambda args: upload_func(*args), file_list))

def download_single(s3_client, bucket, key, local_path):
    """Download a single file from S3 with error handling."""
    try:
        s3_client.download_file(bucket, key, local_path, Config=TRANSFER_CONFIG)
        print(f"Downloaded s3://{bucket}/{key} to {local_path}")
    except ClientError as e:
        print(f"Error downloading {key}: {e}")

def download_directory(args):
    """Download the directory from S3 to local with parallel processing."""
    s3_client = create_s3_client()
    local_dir = args.local_dir
    s3_prefix = args.s3_prefix.rstrip('/') + '/'

    os.makedirs(local_dir, exist_ok=True)

    # Collect all objects to download
    object_list = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        if 'Contents' in page:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not key.endswith('/'):  # Skip directories
                    relative_path = key[len(s3_prefix):]
                    local_path = os.path.join(local_dir, relative_path)
                    object_list.append((key, local_path))

    if not object_list:
        print(f"No objects found under s3://{BUCKET_NAME}/{s3_prefix}")
        return

    # Create all necessary directories first
    for _, local_path in object_list:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading {len(object_list)} files from s3://{BUCKET_NAME}/{s3_prefix}...")

    # Parallel download
    download_func = partial(download_single, s3_client, BUCKET_NAME)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(lambda args: download_func(*args), object_list))

def delete_in_chunks(s3_client, bucket, keys_to_delete):
    """Delete objects in chunks of 1000 to handle large deletions."""
    for i in range(0, len(keys_to_delete), 1000):
        chunk = keys_to_delete[i:i+1000]
        try:
            s3_client.delete_objects(Bucket=bucket, Delete={'Objects': chunk})
        except ClientError as e:
            print(f"Error deleting chunk: {e}")

def delete_prefix(args):
    """Delete all objects under the specified S3 prefix after confirmation."""
    s3_client = create_s3_client()
    s3_prefix = args.s3_prefix.rstrip('/') + '/'

    confirm = input(f"Are you sure you want to delete all data under s3://{BUCKET_NAME}/{s3_prefix}? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Delete operation cancelled.")
        return

    print(f"Deleting all objects under s3://{BUCKET_NAME}/{s3_prefix}...")
    keys_to_delete = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        if 'Contents' in page:
            for obj in page.get('Contents', []):
                keys_to_delete.append({'Key': obj['Key']})

    if keys_to_delete:
        delete_in_chunks(s3_client, BUCKET_NAME, keys_to_delete)
        print(f"Deleted {len(keys_to_delete)} objects from s3://{BUCKET_NAME}/{s3_prefix}")
    else:
        print(f"No objects found under s3://{BUCKET_NAME}/{s3_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync model artifacts with S3.",
        epilog="""
Examples:
  python s3_sync_model.py upload --local_dir output --s3-prefix polyp_data_ml/models/detection/custom_rtdetr_r18vd_polyp/rtdetr_r18vd_6x_detection_v2/
    (Uploads from 'output' to S3 prefix, skipping duplicates)

  python s3_sync_model.py download --local_dir output --s3-prefix polyp_data_ml/models/detection/custom_rtdetr_r18vd_polyp/rtdetr_r18vd_6x_detection_v1/
    (Downloads from S3 prefix to 'output', skipping duplicates)

  python s3_sync_model.py delete --s3-prefix polyp_data_ml/models/detection/custom_rtdetr_r18vd_polyp/rtdetr_r18vd_6x_detection_v1/
    (Deletes all objects under S3 prefix after confirmation)

Note: The script uses the bucket 'seekiq-s3-dev'. Prefixes are created automatically on upload if they don't exist.
        """
    )
    parser.add_argument('mode', choices=['upload', 'download', 'delete'], help="Choose 'upload', 'download', or 'delete'")
    parser.add_argument('--local_dir', type=str, default='output',
                        help="Path to local directory for upload/download (default: 'output')")
    parser.add_argument('--s3-prefix', type=str, default=DEFAULT_PREFIX,
                        help="S3 prefix (default: 'polyp_data_ml/models/detection/custom_rtdetr_r18vd_polyp/')")

    args = parser.parse_args()

    if args.mode == 'upload':
        upload_directory(args)
    elif args.mode == 'download':
        download_directory(args)
    elif args.mode == 'delete':
        delete_prefix(args)