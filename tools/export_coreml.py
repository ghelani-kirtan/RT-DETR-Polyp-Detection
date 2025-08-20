import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np # Needed for TensorType dtype
import coremltools as ct
from src.core import YAMLConfig
from src.zoo.rtdetr import RTDETR

# --- THE SOLUTION: A Wrapper Module ---
class RTDETRWrapper(torch.nn.Module):
    """
    A wrapper to convert the model's dictionary output to a tuple,
    which is required for TorchScript tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, orig_target_sizes):
        # Call the original model
        outputs = self.model(images, orig_target_sizes)
        
        # Unpack the dictionary into a tuple in a fixed order
        # The CoreML output names will correspond to this order.
        return outputs['labels'], outputs['boxes'], outputs['scores']

def main(args):
    """Export to CoreML"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Load checkpoint weights
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)
    print("Checkpoint loaded successfully.")

    # Set the model to evaluation/deploy mode
    model = cfg.model.deploy()
    model.eval()

    # --- Use the Wrapper for Tracing ---
    # Instead of tracing the original model, we trace our new wrapper
    wrapped_model = RTDETRWrapper(model)

    # Dummy inputs for tracing
    dummy_input_images = torch.rand(1, 3, args.input_size, args.input_size)
    # The format must be [[width, height]]
    dummy_input_orig_size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.int64)

    print("Tracing the model with TorchScript...")
    # Trace the wrapped model. It now returns a tuple, which is JIT-compatible.
    traced_model = torch.jit.trace(wrapped_model, (dummy_input_images, dummy_input_orig_size))
    print("Model traced successfully.")

    # Convert to CoreML
    print("Converting to CoreML...")
    ml_model = ct.convert(
        traced_model,
        inputs=[
            # The 'images' input should be an ImageType for direct use with camera frames
            ct.ImageType(name="images", shape=dummy_input_images.shape, scale=1/255.0, bias=[0,0,0]),
            # The 'orig_target_sizes' is still needed for the model's internal scaling logic
            ct.TensorType(name="orig_target_sizes", shape=dummy_input_orig_size.shape, dtype=np.int64)
        ],
        outputs=[
            # These names map directly to the tuple elements returned by the wrapper
            ct.TensorType(name="labels"),
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores")
        ],
        minimum_deployment_target=ct.target.macOS13, # Recommended for modern Macs
        compute_units=ct.ComputeUnit.ALL # Allows the OS to pick the best processor (ANE, GPU, CPU)
    )
    print("Conversion successful.")

    # Save the CoreML model
    ml_model.save(args.output_file)
    print(f"✅ CoreML model saved to {args.output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR model to CoreML format.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='rtdetr_model.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)