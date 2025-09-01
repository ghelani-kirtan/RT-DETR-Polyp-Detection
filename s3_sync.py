import boto3
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# S3:::::
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Constants
BUCKET_NAME = 'seekiq-s3-dev'
S3_PREFIX = 'polyp_data_ml/models/classification/custom_rtdetr_r18vd_polyp/'
LOCAL_DIR = 'output'

# Initialize boto3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=aws_region
                         )
def upload_directory():
    """Upload the local directory to S3."""
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, LOCAL_DIR)
            s3_key = os.path.join(S3_PREFIX, relative_path).replace('\\', '/')
            print(f'Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}')
            s3_client.upload_file(local_path, BUCKET_NAME, s3_key)

def download_directory():
    """Download the directory from S3 to local."""
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                if not s3_key.endswith('/'):  # Skip directories
                    relative_path = os.path.relpath(s3_key, S3_PREFIX)
                    local_path = os.path.join(LOCAL_DIR, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    print(f'Downloading s3://{BUCKET_NAME}/{s3_key} to {local_path}')
                    s3_client.download_file(BUCKET_NAME, s3_key, local_path)

def delete_prefix():
    """Delete all objects under the specified S3 prefix."""
    paginator = s3_client.get_paginator('list_objects_v2')
    delete_keys = {'Objects': []}
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                delete_keys['Objects'].append({'Key': obj['Key']})
                print(f'Marking for deletion: {obj["Key"]}')
    
    if delete_keys['Objects']:
        s3_client.delete_objects(Bucket=BUCKET_NAME, Delete=delete_keys)
        print(f'Deleted {len(delete_keys["Objects"])} objects from s3://{BUCKET_NAME}/{S3_PREFIX}')
    else:
        print(f'No objects found under s3://{BUCKET_NAME}/{S3_PREFIX}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload, download, or delete model artifacts to/from S3.')
    parser.add_argument('mode', choices=['upload', 'download', 'delete'], help='Choose "upload", "download", or "delete"')
    args = parser.parse_args()

    if args.mode == 'upload':
        upload_directory()
    elif args.mode == 'download':
        download_directory()
    elif args.mode == 'delete':
        delete_prefix()