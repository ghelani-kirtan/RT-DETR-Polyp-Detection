"""Core S3 operations manager."""

import os
import boto3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class S3Manager:
    """Manages S3 operations with robust error handling and progress tracking."""
    
    def __init__(self, config: dict):
        """
        Initialize S3 Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bucket = config['s3']['bucket']
        self.region = config['s3'].get('region', 'us-east-1')
        
        # Initialize S3 client
        self.s3_client = self._create_client()
        
        # Transfer configuration
        transfer_cfg = config['transfer']
        self.transfer_config = TransferConfig(
            multipart_threshold=transfer_cfg['multipart_threshold'] * 1024 * 1024,
            multipart_chunksize=transfer_cfg['multipart_chunksize'] * 1024 * 1024,
            max_concurrency=transfer_cfg['max_concurrency'],
            num_download_attempts=transfer_cfg['num_download_attempts'],
            use_threads=True
        )
        
        self.max_workers = transfer_cfg['max_workers']
        self.show_progress = config['display']['show_progress']
        self.verbose = config['display']['verbose']
    
    def _create_client(self):
        """Create boto3 S3 client with credentials."""
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if aws_key and aws_secret:
            session = boto3.Session(
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=self.region
            )
            return session.client('s3')
        else:
            # Use IAM role or default credentials
            return boto3.client('s3', region_name=self.region)
    
    def list_objects(self, prefix: str) -> List[Dict]:
        """
        List all objects under a prefix.
        
        Args:
            prefix: S3 prefix to list
            
        Returns:
            List of object metadata dictionaries
        """
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('/'):  # Skip directories
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified']
                            })
        except ClientError as e:
            print(f"✗ Error listing objects: {e}")
            return []
        
        return objects
    
    def list_prefixes(self, prefix: str) -> List[str]:
        """
        List immediate subdirectories (prefixes) under a prefix.
        
        Args:
            prefix: S3 prefix to list
            
        Returns:
            List of prefix names
        """
        prefixes = []
        
        try:
            result = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' in result:
                for p in result['CommonPrefixes']:
                    prefix_name = p['Prefix'].rstrip('/').split('/')[-1]
                    prefixes.append(prefix_name)
        except ClientError as e:
            print(f"✗ Error listing prefixes: {e}")
            return []
        
        return prefixes
    
    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        Upload a single file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 key (full path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                Config=self.transfer_config
            )
            if self.verbose:
                print(f"✓ Uploaded: {local_path.name}")
            return True
        except ClientError as e:
            print(f"✗ Error uploading {local_path}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download a single file from S3.
        
        Args:
            s3_key: S3 key (full path)
            local_path: Local file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket,
                s3_key,
                str(local_path),
                Config=self.transfer_config
            )
            if self.verbose:
                print(f"✓ Downloaded: {local_path.name}")
            return True
        except ClientError as e:
            print(f"✗ Error downloading {s3_key}: {e}")
            return False
    
    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        exclude_patterns: List[str] = None
    ) -> Tuple[int, int]:
        """
        Upload entire directory to S3 with parallel processing.
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (destination)
            exclude_patterns: Patterns to exclude
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not local_dir.exists():
            print(f"✗ Local directory not found: {local_dir}")
            return 0, 0
        
        # Collect files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(local_dir):
            # Filter out excluded directories
            if exclude_patterns:
                dirs[:] = [d for d in dirs if not self._should_exclude(d, exclude_patterns)]
            
            for file in files:
                if exclude_patterns and self._should_exclude(file, exclude_patterns):
                    continue
                
                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_dir)
                s3_key = f"{s3_prefix.rstrip('/')}/{str(relative_path).replace(os.sep, '/')}"
                files_to_upload.append((local_path, s3_key))
        
        if not files_to_upload:
            print("✗ No files to upload")
            return 0, 0
        
        print(f"📤 Uploading {len(files_to_upload)} files to s3://{self.bucket}/{s3_prefix}")
        
        # Parallel upload with progress bar
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.upload_file, local_path, s3_key): (local_path, s3_key)
                for local_path, s3_key in files_to_upload
            }
            
            if self.show_progress:
                progress = tqdm(total=len(files_to_upload), desc="Uploading", unit="file")
            
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                
                if self.show_progress:
                    progress.update(1)
            
            if self.show_progress:
                progress.close()
        
        return successful, failed
    
    def download_directory(
        self,
        s3_prefix: str,
        local_dir: Path
    ) -> Tuple[int, int]:
        """
        Download entire directory from S3 with parallel processing.
        
        Args:
            s3_prefix: S3 prefix (source)
            local_dir: Local directory path (destination)
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        # List all objects
        objects = self.list_objects(s3_prefix)
        
        if not objects:
            print(f"✗ No objects found at s3://{self.bucket}/{s3_prefix}")
            return 0, 0
        
        print(f"📥 Downloading {len(objects)} files from s3://{self.bucket}/{s3_prefix}")
        
        # Prepare download tasks
        downloads = []
        for obj in objects:
            s3_key = obj['key']
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            local_path = local_dir / relative_path
            downloads.append((s3_key, local_path))
        
        # Parallel download with progress bar
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_file, s3_key, local_path): (s3_key, local_path)
                for s3_key, local_path in downloads
            }
            
            if self.show_progress:
                progress = tqdm(total=len(downloads), desc="Downloading", unit="file")
            
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                
                if self.show_progress:
                    progress.update(1)
            
            if self.show_progress:
                progress.close()
        
        return successful, failed
    
    def delete_prefix(self, s3_prefix: str) -> int:
        """
        Delete all objects under a prefix.
        
        Args:
            s3_prefix: S3 prefix to delete
            
        Returns:
            Number of objects deleted
        """
        objects = self.list_objects(s3_prefix)
        
        if not objects:
            print(f"✗ No objects found at s3://{self.bucket}/{s3_prefix}")
            return 0
        
        print(f"🗑️  Deleting {len(objects)} objects from s3://{self.bucket}/{s3_prefix}")
        
        # Delete in chunks of 1000 (S3 limit)
        deleted = 0
        for i in range(0, len(objects), 1000):
            chunk = objects[i:i+1000]
            keys_to_delete = [{'Key': obj['key']} for obj in chunk]
            
            try:
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': keys_to_delete}
                )
                deleted += len(keys_to_delete)
            except ClientError as e:
                print(f"✗ Error deleting chunk: {e}")
        
        return deleted
    
    def _should_exclude(self, name: str, patterns: List[str]) -> bool:
        """Check if name matches any exclude pattern."""
        from fnmatch import fnmatch
        return any(fnmatch(name, pattern) for pattern in patterns)
    
    def get_object_info(self, s3_key: str) -> Optional[Dict]:
        """
        Get metadata for a single object.
        
        Args:
            s3_key: S3 key
            
        Returns:
            Object metadata or None if not found
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag']
            }
        except ClientError:
            return None
