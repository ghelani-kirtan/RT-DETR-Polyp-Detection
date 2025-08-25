import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Create S3 client
s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                  region_name=AWS_REGION)

# S3 details
bucket_name = 'seekiq-s3-dev'
prefix = 'polyp_data_ml/data/'

# Local directory to save data
local_dir = 'dataset'
os.makedirs(os.path.join(local_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(local_dir, 'masks'), exist_ok=True)

def download_folder(s3_folder, local_folder):
    """
    Download all files from an S3 folder to local.
    """
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix + s3_folder)

    files = []
    for page in pages:
        if 'Contents' in page:
            files.extend(page['Contents'])

    for obj in tqdm(files, desc=f'Downloading {s3_folder}'):
        key = obj['Key']
        if not key.endswith('/'):  # Skip folders
            local_path = os.path.join(local_folder, os.path.basename(key))
            s3.download_file(bucket_name, key, local_path)

try:
    # Download images and masks
    download_folder('images/', os.path.join(local_dir, 'images'))
    download_folder('masks/', os.path.join(local_dir, 'masks'))
    print("Download completed successfully.")
except NoCredentialsError:
    print("Error: AWS credentials not available. Check your .env file.")
except Exception as e:
    print(f"Error during download: {str(e)}")