#!/usr/bin/env python3
"""CLI for classification dataset pipeline."""

import argparse
from pathlib import Path
from .pipelines import ClassificationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Classification dataset pipeline - download, organize, clean, and prepare COCO format"
    )
    
    # Pipeline options
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for all operations (default: current directory)"
    )
    parser.add_argument(
        "--step",
        choices=["download", "organize", "clean", "prepare", "full"],
        default="full",
        help="Pipeline step to run (default: full)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step in full pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without making changes"
    )
    
    # Download options
    parser.add_argument(
        "--api-url",
        type=str,
        help="API URL for dataset versions"
    )
    parser.add_argument(
        "--dataset-version-ids",
        type=int,
        nargs="+",
        help="List of dataset version IDs to download"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(
        base_dir=Path(args.base_dir),
        dataset_version_ids=args.dataset_version_ids,
        api_url=args.api_url,
        dry_run=args.dry_run
    )
    
    # Run requested step
    if args.step == "download":
        pipeline.run_download()
    elif args.step == "organize":
        pipeline.run_organize()
    elif args.step == "clean":
        pipeline.run_clean()
    elif args.step == "prepare":
        pipeline.run_prepare()
    elif args.step == "full":
        pipeline.run_full_pipeline(skip_download=args.skip_download)


if __name__ == "__main__":
    main()
