import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from src.core import YAMLConfig

# This is the final, definitive script for CoreML export,
# mirroring the logic of the official export_onnx.py script.

def main(args):
    """Main export function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Load checkpoint weights into the training-mode model
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    print("Checkpoint loaded successfully.")

    # --- Step 1: Create the Full Pipeline Model (Identical to export_onnx.py) ---
    # This class combines the core model and the post-processor into a single module.
    class FullPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            # This is the full end-to-end inference logic
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            # The output here is a dictionary, which is not traceable by default.
            return outputs

    # --- Step 2: Create a JIT-Traceable Wrapper ---
    # This wrapper's only job is to convert the dictionary output to a tracer-friendly tuple.
    class TraceableWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.pipeline = FullPipeline()

        def forward(self, images, orig_target_sizes):
            output_dict = self.pipeline(images, orig_target_sizes)
            # Unpack the dictionary into a tuple in a fixed, reliable order.
            return output_dict['labels'], output_dict['boxes'], output_dict['scores']

    # Instantiate our final, traceable model
    model_to_trace = TraceableWrapper()
    model_to_trace.eval()

    # --- Step 3: Trace and Convert to CoreML ---
    # Create dummy inputs for tracing
    dummy_images = torch.rand(1, 3, args.input_size, args.input_size)
    # The format must be [[width, height]] with dtype int64
    dummy_size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.int64)

    print("Tracing the full model pipeline with TorchScript...")
    # Trace the wrapper, which now returns a tuple and will succeed.
    traced_model = torch.jit.trace(model_to_trace, (dummy_images, dummy_size))
    print("Model traced successfully.")

    print("Converting to CoreML...")
    ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(name="images", shape=dummy_images.shape),
            ct.TensorType(name="orig_target_sizes", shape=dummy_size.shape, dtype=np.int64)
        ],
        outputs=[
            # These names correspond to the tuple order from TraceableWrapper
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
    ml_model.author = "Exported using official ONNX logic"
    ml_model.save(args.output_file)
    
    print(f"\n✅ SUCCESS: CoreML model saved to {args.output_file}")
    print("This model is self-contained and includes all post-processing.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR model to CoreML using the official pipeline structure.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='rtdetr_full_pipeline.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)