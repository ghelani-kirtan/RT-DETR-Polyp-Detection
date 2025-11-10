"""Command-line interface for S3 tools."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Fix for python -m execution
if __name__ == '__main__' and __package__ is None:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 's3_tools'

from .utils import load_config, confirm_action, build_s3_prefix, print_model_info, print_s3_objects, get_model_summary
from .s3_manager import S3Manager
from .model_detector import ModelDetector


class S3CLI:
    """Command-line interface for S3 operations."""
    
    def __init__(self):
        """Initialize CLI."""
        try:
            self.config = load_config()
        except FileNotFoundError as e:
            print(f"✗ {e}")
            sys.exit(1)
        
        self.s3_manager = S3Manager(self.config)
        self.model_detector = ModelDetector(self.config)
    
    def list_local_models(self):
        """List local models in output directory."""
        print("🔍 Scanning local models...")
        models = self.model_detector.find_models()
        
        if not models:
            print("  No models found in output directory")
            return
        
        print(f"\n📁 Local Models ({get_model_summary(models)}):")
        for i, model in enumerate(models, 1):
            print(f"\n{i}.")
            print_model_info(model)

    def list_s3_models(self, model_type: Optional[str] = None):
        """List models in S3."""
        print("🔍 Scanning S3 models...")
        
        base_prefix = self.config['s3']['base_prefix']
        
        if model_type:
            # List specific type
            type_prefix = self.config['s3']['prefixes'].get(model_type, model_type)
            prefix = f"{base_prefix}/{type_prefix}"
            model_names = self.s3_manager.list_prefixes(prefix)
            
            if not model_names:
                print(f"  No {model_type} models found in S3")
                return
            
            print(f"\n☁️  S3 {model_type.title()} Models:")
            for name in model_names:
                full_prefix = f"{prefix}/{name}"
                objects = self.s3_manager.list_objects(full_prefix)
                print(f"\n  📦 {name}")
                print_s3_objects(objects, full_prefix)
        else:
            # List all types
            all_found = False
            for type_name, type_prefix in self.config['s3']['prefixes'].items():
                prefix = f"{base_prefix}/{type_prefix}"
                model_names = self.s3_manager.list_prefixes(prefix)
                
                if model_names:
                    all_found = True
                    print(f"\n☁️  S3 {type_name.title()} Models:")
                    for name in model_names:
                        full_prefix = f"{prefix}/{name}"
                        objects = self.s3_manager.list_objects(full_prefix)
                        print(f"\n  📦 {name}")
                        print_s3_objects(objects, full_prefix)
            
            if not all_found:
                print("  No models found in S3")

    def upload_model(self, model_name: Optional[str] = None, model_type: Optional[str] = None, dry_run: bool = False):
        """Upload a model to S3."""
        # Auto-detect model if not specified
        if not model_name:
            models = self.model_detector.find_models()
            if not models:
                print("✗ No models found in output directory")
                return
            
            if len(models) == 1:
                model = models[0]
                print(f"📤 Auto-detected model: {model['name']}")
            else:
                print("\n📁 Available models:")
                for i, model in enumerate(models, 1):
                    print(f"\n{i}.")
                    print_model_info(model)
                
                while True:
                    try:
                        choice = input("\nSelect model (number): ").strip()
                        if not choice:
                            print("✗ Upload cancelled")
                            return
                        
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            model = models[idx]
                            break
                        else:
                            print("✗ Invalid selection")
                    except ValueError:
                        print("✗ Please enter a number")
        else:
            # Find specified model
            models = self.model_detector.find_models()
            model = next((m for m in models if m['name'] == model_name), None)
            
            if not model:
                print(f"✗ Model not found: {model_name}")
                return
        
        # Use detected type if not specified
        if not model_type:
            model_type = model['type']
            if model_type == 'unknown':
                print("⚠️  Could not auto-detect model type")
                types = list(self.config['s3']['prefixes'].keys())
                print(f"Available types: {', '.join(types)}")
                
                while True:
                    model_type = input("Enter model type: ").strip().lower()
                    if model_type in types:
                        break
                    print(f"✗ Invalid type. Choose from: {', '.join(types)}")
        
        # Build S3 prefix
        s3_prefix = build_s3_prefix(self.config, model_type, model['name'])
        
        print(f"\n📤 Upload Plan:")
        print(f"  Local:  {model['path']}")
        print(f"  S3:     s3://{self.s3_manager.bucket}/{s3_prefix}")
        print(f"  Type:   {model_type}")
        print(f"  Size:   {self.model_detector.format_size(model['size'])} ({model['file_count']} files)")
        
        if dry_run:
            print("\n🔍 Dry run - no files will be uploaded")
            return
        
        # Confirm upload
        if self.config['display']['confirm_upload']:
            if not confirm_action("\nProceed with upload?", default=True):
                print("✗ Upload cancelled")
                return
        
        # Check if already exists
        existing_objects = self.s3_manager.list_objects(s3_prefix)
        if existing_objects:
            print(f"\n⚠️  Model already exists in S3 ({len(existing_objects)} files)")
            if not confirm_action("Overwrite existing model?", default=False):
                print("✗ Upload cancelled")
                return
        
        # Upload
        exclude_patterns = self.config['detection']['exclude_patterns']
        successful, failed = self.s3_manager.upload_directory(
            model['path'],
            s3_prefix,
            exclude_patterns
        )
        
        if failed == 0:
            print(f"\n✅ Upload complete! {successful} files uploaded")
            print(f"   S3 location: s3://{self.s3_manager.bucket}/{s3_prefix}")
        else:
            print(f"\n⚠️  Upload completed with errors: {successful} successful, {failed} failed")

    def download_model(self, model_name: str, model_type: Optional[str] = None, dry_run: bool = False):
        """Download a model from S3."""
        # If type not specified, search all types
        if not model_type:
            found_prefix = None
            found_type = None
            
            for type_name, type_prefix in self.config['s3']['prefixes'].items():
                base_prefix = self.config['s3']['base_prefix']
                test_prefix = f"{base_prefix}/{type_prefix}/{model_name}"
                
                if self.s3_manager.list_objects(test_prefix):
                    found_prefix = test_prefix
                    found_type = type_name
                    break
            
            if not found_prefix:
                print(f"✗ Model not found in S3: {model_name}")
                print("\nAvailable models:")
                self.list_s3_models()
                return
            
            s3_prefix = found_prefix
            model_type = found_type
        else:
            s3_prefix = build_s3_prefix(self.config, model_type, model_name)
        
        # Check if exists
        objects = self.s3_manager.list_objects(s3_prefix)
        if not objects:
            print(f"✗ Model not found: s3://{self.s3_manager.bucket}/{s3_prefix}")
            return
        
        # Local destination
        download_dir = Path(self.config['local']['download_dir'])
        local_path = download_dir / model_name
        
        print(f"\n📥 Download Plan:")
        print(f"  S3:     s3://{self.s3_manager.bucket}/{s3_prefix}")
        print(f"  Local:  {local_path}")
        print(f"  Type:   {model_type}")
        print(f"  Files:  {len(objects)}")
        
        if dry_run:
            print("\n🔍 Dry run - no files will be downloaded")
            print_s3_objects(objects, s3_prefix)
            return
        
        # Check if local exists
        if local_path.exists():
            print(f"\n⚠️  Local directory already exists: {local_path}")
            if not confirm_action("Overwrite existing directory?", default=False):
                print("✗ Download cancelled")
                return
        
        # Confirm download
        if self.config['display']['confirm_download']:
            if not confirm_action("\nProceed with download?", default=True):
                print("✗ Download cancelled")
                return
        
        # Download
        successful, failed = self.s3_manager.download_directory(s3_prefix, local_path)
        
        if failed == 0:
            print(f"\n✅ Download complete! {successful} files downloaded")
            print(f"   Local location: {local_path}")
        else:
            print(f"\n⚠️  Download completed with errors: {successful} successful, {failed} failed")

    def delete_model(self, model_name: str, model_type: Optional[str] = None, dry_run: bool = False):
        """Delete a model from S3."""
        # If type not specified, search all types
        if not model_type:
            found_prefix = None
            found_type = None
            
            for type_name, type_prefix in self.config['s3']['prefixes'].items():
                base_prefix = self.config['s3']['base_prefix']
                test_prefix = f"{base_prefix}/{type_prefix}/{model_name}"
                
                if self.s3_manager.list_objects(test_prefix):
                    found_prefix = test_prefix
                    found_type = type_name
                    break
            
            if not found_prefix:
                print(f"✗ Model not found in S3: {model_name}")
                return
            
            s3_prefix = found_prefix
            model_type = found_type
        else:
            s3_prefix = build_s3_prefix(self.config, model_type, model_name)
        
        # Check if exists
        objects = self.s3_manager.list_objects(s3_prefix)
        if not objects:
            print(f"✗ Model not found: s3://{self.s3_manager.bucket}/{s3_prefix}")
            return
        
        print(f"\n🗑️  Delete Plan:")
        print(f"  S3:     s3://{self.s3_manager.bucket}/{s3_prefix}")
        print(f"  Type:   {model_type}")
        print(f"  Files:  {len(objects)}")
        
        if dry_run:
            print("\n🔍 Dry run - no files will be deleted")
            print_s3_objects(objects, s3_prefix)
            return
        
        # Confirm deletion
        if self.config['display']['confirm_delete']:
            print(f"\n⚠️  This will permanently delete {len(objects)} files!")
            if not confirm_action("Are you sure?", default=False):
                print("✗ Deletion cancelled")
                return
        
        # Delete
        deleted = self.s3_manager.delete_prefix(s3_prefix)
        
        if deleted > 0:
            print(f"\n✅ Deletion complete! {deleted} files deleted")
        else:
            print(f"\n⚠️  No files were deleted")
    
    def interactive_mode(self):
        """Interactive mode for easy operations."""
        print("\n🎯 S3 Tools - Interactive Mode")
        print("===============================\n")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. List local models")
            print("2. List S3 models")
            print("3. Upload model")
            print("4. Download model")
            print("5. Delete model")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            try:
                if choice == '1':
                    self.list_local_models()
                elif choice == '2':
                    self.list_s3_models()
                elif choice == '3':
                    self.upload_model()
                elif choice == '4':
                    model_name = input("Enter model name to download: ").strip()
                    if model_name:
                        self.download_model(model_name)
                elif choice == '5':
                    model_name = input("Enter model name to delete: ").strip()
                    if model_name:
                        self.delete_model(model_name)
                elif choice == '6':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("✗ Invalid choice")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")



def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="S3 Tools - Robust model sync operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s list                               # List local models
  %(prog)s list --s3                          # List S3 models
  %(prog)s upload                             # Auto-detect and upload latest model
  %(prog)s upload --model r34_v1_3_2          # Upload specific model
  %(prog)s download r34_v1_3_2                # Download model
  %(prog)s delete r34_v1_3_2                  # Delete model
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['list', 'upload', 'download', 'delete'],
        help='Command to execute (omit for interactive mode)'
    )
    
    parser.add_argument(
        'model_name',
        nargs='?',
        help='Model name (for download/delete commands)'
    )
    
    parser.add_argument(
        '--model', '-m',
        help='Specific model to upload'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=['classification', 'detection'],
        help='Model type'
    )
    
    parser.add_argument(
        '--s3',
        action='store_true',
        help='List S3 models (for list command)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    args = parser.parse_args()
    
    try:
        cli = S3CLI()
        
        if not args.command:
            # Interactive mode
            cli.interactive_mode()
        elif args.command == 'list':
            if args.s3:
                cli.list_s3_models(args.type)
            else:
                cli.list_local_models()
        elif args.command == 'upload':
            cli.upload_model(args.model, args.type, args.dry_run)
        elif args.command == 'download':
            if not args.model_name:
                print("✗ Model name required for download")
                sys.exit(1)
            cli.download_model(args.model_name, args.type, args.dry_run)
        elif args.command == 'delete':
            if not args.model_name:
                print("✗ Model name required for delete")
                sys.exit(1)
            cli.delete_model(args.model_name, args.type, args.dry_run)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
