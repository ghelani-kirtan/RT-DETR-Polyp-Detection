import os
import sys
import torch
import numpy as np
import coremltools as ct

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# --- THE DEFINITIVE SOLUTION: A Wrapper Module ---
class ModelWrapper(torch.nn.Module):
    """
    This wrapper intercepts the model's dictionary output and converts it to a tuple.
    This is the key to making the model traceable for CoreML conversion.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # The core model returns a dictionary, e.g., {'pred_logits': ..., 'pred_boxes': ...}
        outputs_dict = self.model(images)
        
        # We explicitly unpack the dictionary into a tuple in a fixed order.
        # This is what the tracer needs.
        return outputs_dict['pred_logits'], outputs_dict['pred_boxes']


def main(args):
    """
    Exports the core RT-DETR model to CoreML using a tracer-safe wrapper.
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Instantiate the base model. We DO NOT call .deploy().
    model = cfg.model

    # Load checkpoint weights
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    # The state dict keys might have a 'model.' prefix, which we need to remove.
    state = {k.replace('model.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("Checkpoint loaded successfully into the core model.")

    # --- Step 1: Wrap the model ---
    # Instead of using the model directly, we use our new wrapper.
    wrapper = ModelWrapper(model)
    wrapper.eval()

    # --- Step 2: Trace the Wrapper ---
    dummy_input_images = torch.rand(1, 3, args.input_size, args.input_size)
    print("Tracing the wrapped model with TorchScript...")
    
    # This will now succeed because the wrapper returns a tuple.
    traced_model = torch.jit.trace(wrapper, dummy_input_images)
    print("Model traced successfully.")

    # --- Step 3: Convert to CoreML ---
    print("Converting to CoreML...")
    ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(name="images", shape=dummy_input_images.shape)
        ],
        outputs=[
            # The output names correspond to the tuple order in the wrapper
            ct.TensorType(name="pred_logits"), # Shape: [1, num_queries, num_classes]
            ct.TensorType(name="pred_boxes")   # Shape: [1, num_queries, 4]
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    print("Conversion successful.")
    
    ml_model.short_description = "RT-DETR Core Model for Polyp Detection"
    ml_model.save(args.output_file)
    
    print(f"\n✅ SUCCESS: CoreML model saved to {args.output_file}")
    print("\nIMPORTANT: Remember to use the post-processing logic in your Mac app to decode the raw 'pred_logits' and 'pred_boxes' outputs.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR Core model to CoreML format.")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_file', '-o', type=str, default='rtdetr_core_model.mlpackage', help="Path to save the output CoreML model package.")
    parser.add_argument('--input_size', '-s', type=int, default=640, help="Input image size (square).")
    
    args = parser.parse_args()
    main(args)