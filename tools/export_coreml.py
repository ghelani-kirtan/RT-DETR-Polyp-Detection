import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from src.core import YAMLConfig

# This is the final, definitive script for CoreML export.
# It uses torch.jit.script() to correctly handle the model's internal control flow,
# which was the root cause of all previous failures.

def main(args):
    """Main export function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Load checkpoint weights into the training-mode model
    print(f"Loading checkpoint from: {args.resume}")
    # Using weights_only=True is safer and silences the warning.
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    print("Checkpoint loaded successfully.")

    # --- Step 1: Define the Full End-to-End Pipeline ---
    # This class, inspired by the official export_onnx.py, combines the
    # core model and the post-processor into a single module.
    class FullPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            # The postprocessor returns a dictionary, which we will handle.
            processed_outputs = self.postprocessor(outputs, orig_target_sizes)
            # We explicitly unpack the dictionary into a tuple for CoreML compatibility.
            return processed_outputs['labels'], processed_outputs['boxes'], processed_outputs['scores']

    # Instantiate the full pipeline
    pipeline = FullPipeline()
    pipeline.eval()

    # --- Step 2: Compile the Pipeline using torch.jit.script ---
    # This is the CRITICAL FIX. We use .script() instead of .trace().
    # .script() analyzes the Python code and correctly captures the control flow
    # (like 'if' statements) that caused all previous errors.
    print("Compiling the full model pipeline with torch.jit.script...")
    scripted_model = torch.jit.script(pipeline)
    print("Model compiled successfully.")

    # --- Step 3: Convert the Scripted Model to CoreML ---
    # Create dummy inputs for the conversion process
    dummy_images = torch.rand(1, 3, args.input_size, args.input_size)
    dummy_size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.int64)

    print("Converting to CoreML...")
    # We pass the successfully compiled scripted_model to the converter.
    ml_model = ct.convert(
        scripted_model,
        inputs=[
            ct.ImageType(name="images", shape=dummy_images.shape),
            ct.TensorType(name="orig_target_sizes", shape=dummy_size.shape, dtype=np.int64)
        ],
        outputs=[
            # These names correspond to the tuple returned by FullPipeline's forward method
            ct.TensorType(name="labels"),
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores")
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    print("Conversion successful.")

    # Add metadata and save
    ml_model.short_description = "RT-DETR Polyp Detector (Full Pipeline)"
    ml_model.author = "Exported using a robust, script-based method"
    ml_model.save(args.output_file)
    
    print(f"\n✅✅✅ SUCCESS: CoreML model saved to {args.output_file}")
    print("This model is fully self-contained and ready for deployment.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR model to CoreML using torch.jit.script for robustness.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='polyp_detector.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)