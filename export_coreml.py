import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

# Make sure the 'src' directory is in the Python path
# Add the root of your project to the path if necessary
# sys.path.append('/path/to/your/RT-DETR-Polyp-Detection') 
from src.core import YAMLConfig

print(f"Using torch version {torch.__version__}")
print(f"Using coremltools version {ct.__version__}")


def main(args):
    """
    Loads a PyTorch .pth checkpoint, wraps it for inference, and converts it
    to a CoreML .mlpackage file.
    """
    print("Loading configuration from:", args.config)
    cfg = YAMLConfig(args.config)

    print("Loading checkpoint from:", args.resume)
    try:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # Handle both standard and EMA checkpoints
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
            print("Using EMA weights.")
        else:
            state = checkpoint['model']
            print("Using standard model weights.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 1. Recreate the exact inference model structure from the reference script
    # This is crucial as it includes both the model and the post-processing logic.
    class InferenceWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            # Instantiate the model and post-processor from your project's config
            self.model = cfg.model
            self.postprocessor = cfg.postprocessor
            
            # Load the trained weights into the model
            self.model.load_state_dict(state)

            # Set the model to evaluation mode
            self.model.eval()
            
            # Get the deployable versions
            self.model = self.model.deploy()
            self.postprocessor = self.postprocessor.deploy()
            print("Model and post-processor are in deploy mode.")

        def forward(self, images, orig_target_sizes):
            # The exact forward pass for inference
            outputs = self.model(images)
            # The post-processor converts raw output to boxes, labels, scores
            return self.postprocessor(outputs, orig_target_sizes)

    # Instantiate the complete inference model
    model = InferenceWrapper()
    model.eval()

    # 2. Prepare example inputs for tracing
    # The shapes and data types MUST match what the model expects
    print("Preparing example inputs for tracing...")
    example_image = torch.rand(1, 3, 640, 640) # Shape: (batch, channels, height, width)

    # --- THIS IS THE KEY FIX ---
    # The CoreML model requires the size tensor to be float32, not int64.
    # We define it explicitly here. Shape is (batch, 2) for [height, width].
    example_size = torch.tensor([[640., 640.]], dtype=torch.float32) 
    
    print(f"Image input shape: {example_image.shape}, dtype: {example_image.dtype}")
    print(f"Size input shape: {example_size.shape}, dtype: {example_size.dtype}")

    # 3. Trace the model using torch.jit.trace
    # This creates a static graph that CoreML tools can understand.
    try:
        print("Tracing the model with JIT...")
        traced_model = torch.jit.trace(model, (example_image, example_size))
    except Exception as e:
        print(f"Error during model tracing: {e}")
        sys.exit(1)

    # 4. Convert the traced model to CoreML
    print("Starting CoreML conversion...")
    try:
        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",  # Modern, flexible format
            minimum_deployment_target=ct.target.macOS12, # or iOS15, etc.
            inputs=[
                # Define the image input with its name and shape
                ct.ImageType(
                    name="images", # This name must match the forward() argument
                    shape=example_image.shape,
                    scale=1/255.0, # Normalizes pixel values from [0,255] to [0,1]
                    color_layout=ct.colorlayout.RGB,
                    channel_first=True
                ),
                # Define the size input with its name, shape, and CORRECT data type
                ct.TensorType(
                    name="orig_target_sizes", # Must match the forward() argument
                    shape=example_size.shape,
                    dtype=np.float32 # Explicitly set the data type to float32
                )
            ],
            # Optional: You can name the outputs if you know them
            # outputs=[ct.TensorType(name="labels"), ct.TensorType(name="boxes"), ct.TensorType(name="scores")]
        )

        # 5. Save the final CoreML model package
        print(f"Conversion successful! Saving model to {args.output}")
        coreml_model.save(args.output)
        print("Done.")

    except Exception as e:
        print(f"\n--- CoreML Conversion Failed ---")
        print(f"An error occurred during the conversion process: {e}")
        print("This often happens if the model contains an operation not supported by CoreML.")
        print("Check the error logs above for details on the failing operator.")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert RT-DETR PyTorch model to CoreML")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model's YAML configuration file.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the .pth model checkpoint file.")
    parser.add_argument('-o', '--output', type=str, default="rtdetr_polyp.mlpackage", help="Path to save the output CoreML model package.")
    args = parser.parse_args()
    main(args)