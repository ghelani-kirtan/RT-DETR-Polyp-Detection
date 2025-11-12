#!/usr/bin/env python3
"""
Compare Detection Preparer V1 vs V2.

This script runs both preparers on the same dataset and compares results.
"""

import json
import argparse
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline


def compare_preparers(base_dir: str, enable_subset_removal: bool = False):
    """
    Compare V1 and V2 preparers.
    
    Args:
        base_dir: Base directory containing detection_dataset
        enable_subset_removal: Enable subset removal in V2
    """
    base_path = Path(base_dir)
    
    print("=" * 70)
    print("🔬 Detection Preparer Comparison: V1 vs V2")
    print("=" * 70)
    
    # Run V1 (Original)
    print("\n📦 Running V1 (Original Preparer)...")
    print("-" * 70)
    
    pipeline_v1 = DetectionPipeline(
        base_dir=base_path,
        use_enhanced_preparer=False
    )
    pipeline_v1.coco_dir = base_path / "coco_v1"
    pipeline_v1.preparer_config.output_dir = pipeline_v1.coco_dir
    
    stats_v1 = pipeline_v1.run_prepare()
    
    # Run V2 (Enhanced)
    print("\n📦 Running V2 (Enhanced Preparer)...")
    print("-" * 70)
    
    pipeline_v2 = DetectionPipeline(
        base_dir=base_path,
        use_enhanced_preparer=True,
        min_area_percentage=0.0005,
        remove_subset_boxes=enable_subset_removal,
        subset_threshold=0.85,
        parallel_workers=8
    )
    pipeline_v2.coco_dir = base_path / "coco_v2"
    pipeline_v2.preparer_config.output_dir = pipeline_v2.coco_dir
    
    stats_v2 = pipeline_v2.run_prepare()
    
    # Compare results
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    # Extract stats
    v1_train = stats_v1.get('train', {})
    v1_val = stats_v1.get('val', {})
    v2_train = stats_v2.get('train', {})
    v2_val = stats_v2.get('val', {})
    
    v1_total_images = v1_train.get('images', 0) + v1_val.get('images', 0)
    v2_total_images = v2_train.get('images', 0) + v2_val.get('images', 0)
    
    v1_total_ann = v1_train.get('annotations', 0) + v1_val.get('annotations', 0)
    v2_total_ann = v2_train.get('annotations', 0) + v2_val.get('annotations', 0)
    
    # Print comparison table
    print(f"\n{'Metric':<30} {'V1 (Original)':<20} {'V2 (Enhanced)':<20} {'Difference':<15}")
    print("-" * 85)
    
    # Images
    print(f"{'Total Images':<30} {v1_total_images:<20} {v2_total_images:<20} {v2_total_images - v1_total_images:<15}")
    print(f"{'  Train Images':<30} {v1_train.get('images', 0):<20} {v2_train.get('images', 0):<20} {'-':<15}")
    print(f"{'  Val Images':<30} {v1_val.get('images', 0):<20} {v2_val.get('images', 0):<20} {'-':<15}")
    
    # Annotations
    ann_diff = v2_total_ann - v1_total_ann
    print(f"\n{'Total Annotations':<30} {v1_total_ann:<20} {v2_total_ann:<20} {ann_diff:<15}")
    print(f"{'  Train Annotations':<30} {v1_train.get('annotations', 0):<20} {v2_train.get('annotations', 0):<20} {'-':<15}")
    print(f"{'  Val Annotations':<30} {v1_val.get('annotations', 0):<20} {v2_val.get('annotations', 0):<20} {'-':<15}")
    
    # Averages
    v1_avg = v1_total_ann / v1_total_images if v1_total_images > 0 else 0
    v2_avg = v2_total_ann / v2_total_images if v2_total_images > 0 else 0
    print(f"\n{'Avg Annotations/Image':<30} {v1_avg:<20.2f} {v2_avg:<20.2f} {v2_avg - v1_avg:<15.2f}")
    
    # V2 Enhanced Stats
    if 'enhanced' in stats_v2:
        enhanced = stats_v2['enhanced']
        print(f"\n{'V2 Enhanced Filtering:':<30}")
        print(f"{'  Small Boxes Filtered':<30} {'-':<20} {enhanced.get('small_boxes_filtered', 0):<20} {'-':<15}")
        print(f"{'  Subset Boxes Removed':<30} {'-':<20} {enhanced.get('subset_boxes_removed', 0):<20} {'-':<15}")
        
        total_filtered = enhanced.get('small_boxes_filtered', 0) + enhanced.get('subset_boxes_removed', 0)
        print(f"{'  Total Filtered':<30} {'-':<20} {total_filtered:<20} {'-':<15}")
    
    # Percentage change
    if v1_total_ann > 0:
        pct_change = ((v2_total_ann - v1_total_ann) / v1_total_ann) * 100
        print(f"\n📈 Annotation Change: {pct_change:+.2f}%")
        
        if pct_change < -1:
            print(f"✅ V2 filtered {abs(ann_diff)} boxes ({abs(pct_change):.1f}% reduction)")
            print("   This suggests V2 removed noise/duplicates")
        elif pct_change > 1:
            print(f"⚠️  V2 added {ann_diff} boxes ({pct_change:.1f}% increase)")
            print("   This is unexpected - review V2 settings")
        else:
            print("➡️  Minimal change in annotation count")
            print("   Both preparers produced similar results")
    
    # Output locations
    print("\n" + "=" * 70)
    print("📁 Output Directories:")
    print(f"  V1: {pipeline_v1.coco_dir}")
    print(f"  V2: {pipeline_v2.coco_dir}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if ann_diff < -10:
        print("  ✅ V2 shows significant filtering - review visualizations")
        print("  ✅ If results look good, consider switching to V2")
    elif ann_diff > 10:
        print("  ⚠️  V2 produced more annotations - investigate why")
        print("  ⚠️  Check V2 configuration settings")
    else:
        print("  ➡️  Similar results - either preparer should work")
        print("  ➡️  V2 offers more control if needed later")
    
    print("\n📝 Next Steps:")
    print("  1. Review annotation files in both output directories")
    print("  2. Visualize some samples to compare quality")
    print("  3. Check the enhanced stats in V2 output")
    print("  4. Decide which preparer to use for production")
    
    # Save comparison report
    report = {
        'v1': stats_v1,
        'v2': stats_v2,
        'comparison': {
            'total_images': {
                'v1': v1_total_images,
                'v2': v2_total_images,
                'difference': v2_total_images - v1_total_images
            },
            'total_annotations': {
                'v1': v1_total_ann,
                'v2': v2_total_ann,
                'difference': ann_diff,
                'percentage_change': pct_change if v1_total_ann > 0 else 0
            },
            'avg_per_image': {
                'v1': v1_avg,
                'v2': v2_avg,
                'difference': v2_avg - v1_avg
            }
        }
    }
    
    report_path = base_path / "preparer_comparison.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print("=" * 70)
    
    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare Detection Preparer V1 vs V2"
    )
    parser.add_argument(
        "--base-dir",
        default="data/temp",
        help="Base directory containing detection_dataset (default: data/temp)"
    )
    parser.add_argument(
        "--enable-subset-removal",
        action="store_true",
        help="Enable subset removal in V2 (default: disabled)"
    )
    
    args = parser.parse_args()
    
    try:
        report = compare_preparers(
            args.base_dir,
            args.enable_subset_removal
        )
        print("\n✅ Comparison complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
