"""Direct S3 downloader."""

from pathlib import Path
from typing import Dict
from tqdm import tqdm
from .base_downloader import BaseDownloader
from ..core import DownloaderConfig, S3Utils


class S3Downloader(BaseDownloader):
    """Download data directly from S3 bucket."""
    
    def __init__(self, config: DownloaderConfig):
        """Initialize S3 downloader."""
        super().__init__(config)
        self.s3_utils = S3Utils(
            max_pool_connections=config.max_pool_connections,
            retry_attempts=config.retry_attempts
        )
    
    def download(self) -> Dict[str, any]:
        """
        Download data from S3 bucket.
        
        Returns:
            Download statistics
        """
        stats = {"total_files": 0, "failed": 0}
        
        # List objects in bucket
        objects = self.s3_utils.list_objects(
            self.config.bucket_name,
            self.config.s3_prefix
        )
        
        self.logger.info(f"Found {len(objects)} objects to download")
        
        for obj_key in tqdm(objects, desc="Downloading from S3"):
            if obj_key.endswith('/'):
                continue
            
            # Determine local path based on S3 key structure
            relative_path = obj_key.replace(self.config.s3_prefix, '').lstrip('/')
            local_path = self.config.output_dir / relative_path
            
            success = self.s3_utils.download_file(
                self.config.bucket_name,
                obj_key,
                local_path,
                self.config.dry_run
            )
            
            if success:
                stats["total_files"] += 1
            else:
                stats["failed"] += 1
                self.logger.error(f"Failed to download: {obj_key}")
        
        return stats
    
    def verify_downloads(self) -> Dict[str, any]:
        """
        Verify downloaded files exist.
        
        Returns:
            Verification results
        """
        # Simple verification - check if output directory has files
        if self.config.output_dir.exists():
            file_count = sum(1 for _ in self.config.output_dir.rglob('*') if _.is_file())
            return {"files_found": file_count}
        return {"files_found": 0}
