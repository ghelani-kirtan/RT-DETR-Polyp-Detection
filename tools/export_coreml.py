import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from src.core import YAMLConfig
# We must import the specific block that causes the error to patch it.
from src.nn.backbone.presnet import BasicBlock

# This is the final, definitive script for CoreML export.
# It includes a critical patching function to make the model compatible with torch.jit.script().

def patch_presnet_for_scripting(model: nn.Module):
    """
    Finds all BasicBlock modules within the model and replaces the problematic
    `self.short = None` with a script-friendly `nn.Identity()` module.
    This is the key fix for the JIT compilation error.
    """
    print("Applying patch to PResNet backbone for JIT compatibility...")
    count = 0
    for module in model.modules():
        if isinstance(module, BasicBlock):
            if not hasattr(module, 'short') or module.short is None:
                module.short = nn.Identity()
                count += 1
    if count > 0:
        print(f"Successfully patched {count} BasicBlock instances.")
    else:
        print("No BasicBlock instances needed patching.")


def main(args):
    """Main export function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Load checkpoint weights
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    print("Checkpoint loaded successfully.")

    # Define the Full End-to-End Pipeline, same as the official ONNX exporter
    class FullPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            processed_outputs = self.postprocessor(outputs, orig_target_sizes)
            return processed_outputs['labels'], processed_outputs['boxes'], processed_outputs['scores']

    # Instantiate the full pipeline
    pipeline = FullPipeline()
    pipeline.eval()

    # --- THE CRITICAL FIX ---
    # Apply the in-memory patch to the model before attempting to script it.
    patch_presnet_for_scripting(pipeline)

    # Compile the patched pipeline using torch.jit.script
    print("Compiling the patched model pipeline with torch.jit.script...")
    scripted_model = torch.jit.script(pipeline)
    print("Model compiled successfully.")

    # Create dummy inputs for the conversion process
    dummy_images = torch.rand(1, 3, args.input_size, args.input_size)
    dummy_size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.int64)

    # Convert the scripted model to CoreML
    print("Converting to CoreML...")
    ml_model = ct.convert(
        scripted_model,
        inputs=[
            ct.ImageType(name="images", shape=dummy_images.shape),
            ct.TensorType(name="orig_target_sizes", shape=dummy_size.shape, dtype=np.int64)
        ],
        outputs=[
            ct.TensorType(name="labels"),
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores")
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    print("Conversion successful.")

    # Add metadata and save
    ml_model.short_description = "RT-DETR Polyp Detector (Full Pipeline, Patched)"
    ml_model.save(args.output_file)
    
    print(f"\n✅✅✅ SUCCESS: CoreML model saved to {args.output_file}")
    print("This model is fully self-contained and ready for deployment.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR model to CoreML using a robust, patching-based method.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='polyp_detector.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)