import os
import sys
import torch
import numpy as np
import coremltools as ct

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

def main(args):
    """
    Exports the core RT-DETR model (without the post-processor) to CoreML.
    This is the robust and correct method, avoiding tracing issues.
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    # --- Step 1: Instantiate the Core Model ---
    # We use the base model directly from the config.
    # CRUCIALLY, we DO NOT call model.deploy(), as this attaches the
    # non-traceable post-processing module.
    model = cfg.model
    model.eval()

    # Load checkpoint weights
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    # The state dict keys might have a 'model.' prefix, which we need to remove
    # for the base model.
    state = {k.replace('model.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    print("Checkpoint loaded successfully into the core model.")

    # --- Step 2: Trace the Core Model ---
    # The input is just the image tensor. We no longer need 'orig_target_sizes'
    # because that is only used by the post-processor we removed.
    dummy_input_images = torch.rand(1, 3, args.input_size, args.input_size)

    print("Tracing the core model with TorchScript...")
    # This will now succeed because the model is a simple, static graph.
    traced_model = torch.jit.trace(model, dummy_input_images)
    print("Core model traced successfully.")

    # --- Step 3: Convert to CoreML ---
    print("Converting to CoreML...")
    # The model now outputs raw logits and boxes, which we will process manually.
    ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(name="images", shape=dummy_input_images.shape)
        ],
        outputs=[
            # These are the raw outputs from the model's head
            ct.TensorType(name="pred_logits"), # Shape: [1, num_queries, num_classes]
            ct.TensorType(name="pred_boxes")   # Shape: [1, num_queries, 4]
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    print("Conversion successful.")
    
    # Add metadata for clarity
    ml_model.short_description = "RT-DETR Core Model for Polyp Detection"
    ml_model.author = "Exported using custom script"
    ml_model.license = "Specify your license"

    # Save the CoreML model
    ml_model.save(args.output_file)
    print(f"✅ CoreML model saved to {args.output_file}")
    print("\nIMPORTANT: You must now use the provided post-processing logic in your application.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR Core model to CoreML format.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='rtdetr_core_model.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)