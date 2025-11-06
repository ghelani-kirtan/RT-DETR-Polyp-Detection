"""API-based data downloader."""

import requests
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .base_downloader import BaseDownloader
from ..core import DownloaderConfig, S3Utils


class APIDownloader(BaseDownloader):
    """Download data from API with S3 integration."""
    
    def __init__(self, config: DownloaderConfig):
        """Initialize API downloader."""
        super().__init__(config)
        self.s3_utils = S3Utils(
            max_pool_connections=config.max_pool_connections,
            retry_attempts=config.retry_attempts,
            multipart_threshold_mb=config.multipart_threshold_mb,
            multipart_chunksize_mb=config.multipart_chunksize_mb,
            max_concurrency=config.max_concurrency
        )
        self.download_tasks = []
    
    def fetch_dataset_versions(self) -> List[Dict]:
        """
        Fetch dataset versions from API.
        
        Returns:
            List of dataset version data
        """
        body = {
            "dataset_version_identifiers": self.config.dataset_version_ids
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                json=body,
                headers=headers
            )
            response.raise_for_status()
            resp_json = response.json()
            
            if not resp_json.get("success"):
                raise ValueError(f"API error: {resp_json.get('message')}")
            
            return resp_json.get("data", [])
        except Exception as e:
            self.logger.error(f"Error fetching from API: {e}")
            return []
    
    def prepare_download_tasks(
        self,
        frame: Dict,
        video_name: str,
        output_path: Path,
        class_mappings: Dict[str, str]
    ) -> List[Dict]:
        """
        Prepare download tasks for a frame.
        
        Args:
            frame: Frame data from API
            video_name: Video name for prefixing
            output_path: Output directory path
            class_mappings: Mapping of pathology names
            
        Returns:
            List of download task dictionaries
        """
        tasks = []
        frame_id = frame.get("id")
        
        if frame.get("review_status") != "reviewed":
            return tasks
        
        option = frame.get("option", "").lower()
        pathology_raw = (frame.get("pathology") or "").lower().replace(" ", "_")
        pathology = class_mappings.get(pathology_raw)
        image_url = frame.get("image_url")
        mask_url = frame.get("mask_url")
        
        if not image_url:
            return tasks
        
        try:
            bucket, image_key = S3Utils.parse_s3_url(image_url)
            base_filename = Path(image_key).name
            unique_filename = f"{video_name}_{base_filename}"
            
            # Determine if negative or positive sample
            if option != "polyp" or pathology_raw == "no_pathology" or not mask_url:
                # Negative sample
                neg_dir = output_path / "negative_samples"
                tasks.append({
                    "bucket": bucket,
                    "key": image_key,
                    "local_path": neg_dir / unique_filename,
                    "is_mask": False,
                    "frame_id": frame_id,
                    "category": "negative"
                })
            else:
                # Positive sample
                if not pathology:
                    return tasks
                
                pos_dir = output_path / "positive_samples" / pathology
                img_dir = pos_dir / "images"
                mask_dir = pos_dir / "masks"
                
                tasks.append({
                    "bucket": bucket,
                    "key": image_key,
                    "local_path": img_dir / unique_filename,
                    "is_mask": False,
                    "frame_id": frame_id,
                    "category": pathology
                })
                
                if mask_url:
                    _, mask_key = S3Utils.parse_s3_url(mask_url)
                    mask_filename = f"{video_name}_{Path(mask_key).name}"
                    tasks.append({
                        "bucket": bucket,
                        "key": mask_key,
                        "local_path": mask_dir / mask_filename,
                        "is_mask": True,
                        "frame_id": frame_id,
                        "category": pathology
                    })
        except Exception as e:
            self.logger.error(f"Error preparing tasks for frame {frame_id}: {e}")
        
        return tasks
    
    def download(self) -> Dict[str, any]:
        """
        Download data from API and S3.
        
        Returns:
            Download statistics
        """
        dataset_versions = self.fetch_dataset_versions()
        if not dataset_versions:
            self.logger.info("No dataset versions found")
            return {}
        
        stats = {"total_images": 0, "total_masks": 0, "failed": 0}
        
        for version in tqdm(dataset_versions, desc="Processing versions"):
            version_id = version.get("id")
            version_title = version.get(
                "dataset_version_title",
                f"version_{version_id}"
            ).replace(" ", "_")
            
            # ROOT FIX: Put files directly in client_data for simpler pipeline flow
            # If you need version separation, use different base_dir for each version
            output_path = self.config.output_dir / "client_data"
            
            self.prepare_output_structure(output_path)
            
            # Collect download tasks
            self.download_tasks = []
            video_folders = version.get("video_folders", [])
            
            for video in video_folders:
                video_name = video.get(
                    "video_name",
                    f"video_{video.get('video_folder_id')}"
                )
                frames = video.get("frames", [])
                
                for frame in frames:
                    tasks = self.prepare_download_tasks(
                        frame,
                        video_name,
                        output_path,
                        self.config.class_mappings
                    )
                    self.download_tasks.extend(tasks)
            
            # Download in parallel
            if self.download_tasks:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_task = {
                        executor.submit(
                            self.s3_utils.download_file,
                            task["bucket"],
                            task["key"],
                            task["local_path"],
                            self.config.dry_run
                        ): task
                        for task in self.download_tasks
                    }
                    
                    for future in tqdm(
                        as_completed(future_to_task),
                        total=len(self.download_tasks),
                        desc="Downloading files"
                    ):
                        task = future_to_task[future]
                        try:
                            success = future.result()
                            if success:
                                if task["is_mask"]:
                                    stats["total_masks"] += 1
                                else:
                                    stats["total_images"] += 1
                            else:
                                stats["failed"] += 1
                        except Exception as e:
                            self.logger.error(f"Download failed: {e}")
                            stats["failed"] += 1
        
        return stats
    
    def verify_downloads(self) -> Dict[str, any]:
        """
        Verify downloaded files.
        
        Returns:
            Verification results
        """
        missing = []
        for task in self.download_tasks:
            if not task["local_path"].exists():
                missing.append(task)
        
        return {
            "total_expected": len(self.download_tasks),
            "missing_count": len(missing),
            "missing_files": missing
        }
