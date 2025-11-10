"""S3 utility functions for data pipelines."""

import time
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse, unquote
import boto3
import botocore
from boto3.s3.transfer import TransferConfig


class S3Utils:
    """Utility class for S3 operations."""
    
    def __init__(
        self,
        max_pool_connections: int = 200,
        retry_attempts: int = 15,
        multipart_threshold_mb: int = 25,
        multipart_chunksize_mb: int = 25,
        max_concurrency: int = 10
    ):
        """
        Initialize S3 utilities.
        
        Args:
            max_pool_connections: Max connection pool size
            retry_attempts: Max retry attempts
            multipart_threshold_mb: Threshold for multipart downloads
            multipart_chunksize_mb: Chunk size for multipart
            max_concurrency: Max threads for boto3 transfer
        """
        client_config = botocore.config.Config(
            max_pool_connections=max_pool_connections,
            retries={'max_attempts': retry_attempts, 'mode': 'adaptive'}
        )
        self.client = boto3.client('s3', config=client_config)
        
        self.transfer_config = TransferConfig(
            multipart_threshold=1024 * 1024 * multipart_threshold_mb,
            max_concurrency=max_concurrency,
            multipart_chunksize=1024 * 1024 * multipart_chunksize_mb,
            use_threads=True
        )
        
        self.retry_attempts = retry_attempts
    
    @staticmethod
    def parse_s3_url(url: str) -> Tuple[str, str]:
        """
        Parse S3 URL to extract bucket and key.
        
        Args:
            url: S3 URL (presigned or standard)
            
        Returns:
            Tuple of (bucket, key)
        """
        parsed = urlparse(url)
        if not parsed.netloc.endswith('.s3.amazonaws.com'):
            raise ValueError(f"Invalid S3 URL: {url}")
        bucket = parsed.netloc.split('.')[0]
        encoded_key = parsed.path.lstrip('/')
        key = unquote(encoded_key)
        return bucket, key
    
    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: Path,
        dry_run: bool = False,
        retry_delay: int = 5
    ) -> bool:
        """
        Download file from S3 with retries.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local path to save file
            dry_run: If True, simulate download
            retry_delay: Seconds to wait between retries
            
        Returns:
            True if successful, False otherwise
        """
        if dry_run:
            return True
        
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=str(local_path),
                    Config=self.transfer_config
                )
                if local_path.exists() and local_path.stat().st_size > 0:
                    return True
            except Exception:
                pass
            attempt += 1
            time.sleep(retry_delay * attempt)
        return False
    
    def list_objects(self, bucket: str, prefix: str) -> list:
        """
        List objects in S3 bucket with prefix.
        
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
            
        Returns:
            List of object keys
        """
        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        objects = []
        for page in pages:
            if 'Contents' in page:
                objects.extend([obj['Key'] for obj in page['Contents']])
        return objects
