"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import sys
import os 
import sys 
# Add the root directory to the Python path to allow importing from 'src'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

import numpy as np

import torch.nn as nn 
import coremltools as ct
import types # Import the 'types' module for monkey-patching

from src.core import YAMLConfig

import coremltools as ct
from coremltools import ComputeUnit, TensorType, RangeDim

def main(args):
    """main
    """
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)
    else:
        print('not load model.state_dict, use default init state dict...')
    # Load the trained model weights from the checkpoint
    if args.resume:
        # Set weights_only=True to address the FutureWarning and enhance security
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        raise ValueError("--resume argument is required to load trained model weights.")

    class Model(nn.Module):
        def __init__(self, ) -> None:
    # Define a wrapper class for deployment.
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
           
        def forward(self, images, orig_target_sizes):
            
        def forward(self, images):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().eval()  # Set to eval mode

    data = torch.rand(1, 3, args.input_size, args.input_size)
    size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.float32)  # CoreML expects float32 for inputs

    # Trace the model
    traced_model = torch.jit.trace(model, (data, size))

    # Define input shapes
    batch_dim = RangeDim(lower_bound=1, upper_bound=1, default=1) if not args.dynamic else RangeDim(lower_bound=1, upper_bound=8, default=1)
    images_shape = (batch_dim, 3, args.input_size, args.input_size)
    sizes_shape = (batch_dim, 2)

    images_input = TensorType(name='images', shape=images_shape, dtype=np.float32)
    sizes_input = TensorType(name='orig_target_sizes', shape=sizes_shape, dtype=np.float32)

    # Convert to CoreML
    ct_model = ct.convert(
        traced_model,
        convert_to='mlprogram',  # Use mlprogram for better compatibility on M series
        inputs=[images_input, sizes_input],
        outputs=[
            ct.TensorType(name='labels', dtype=np.int32),
            ct.TensorType(name='boxes', dtype=np.float32),
            ct.TensorType(name='scores', dtype=np.float32)
        ],
        compute_units=ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13  # For flexible shapes and mlprogram
    )

    # Optional: Set metadata for object detector preview in Xcode
    # Assuming single class 'polyp'; adjust as needed
    labels = ['background', 'polyp']  # Example; user should modify if multiple classes
    params = {"labels": labels, "coordinates": "x,y,width,height", "grid_anchor_variations": "0,0", "grid_prior_variations": "1,1", "confidence_threshold": 0.25, "iou_threshold": 0.45}
    ct_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "objectDetector"
    ct_model.user_defined_metadata['com.apple.coreml.model.preview.params'] = np.compat.unicode(json.dumps(params))

    # Save the model
    ct_model.save(args.output_file)
    print(f'Exported CoreML model to {args.output_file}')

    if args.check:
        # Test prediction with random input
        import numpy as np
        test_images = np.random.rand(1, 3, args.input_size, args.input_size).astype(np.float32)
        test_sizes = np.array([[args.input_size, args.input_size]], dtype=np.float32)
        predictions = ct_model.predict({'images': test_images, 'orig_target_sizes': test_sizes})
        print('Check export CoreML model done...')
        print('Sample output shapes:')
        print(f"labels: {predictions['labels'].shape}")
        print(f"boxes: {predictions['boxes'].shape}")
        print(f"scores: {predictions['scores'].shape}")
            orig_target_sizes = torch.tensor([[args.input_size, args.input_size]], dtype=torch.float32)
            return self.postprocessor(outputs, orig_target_sizes)

    # Instantiate the model
    model = Model()
    model.eval()

    # --- MONKEY-PATCH FOR COREML COMPATIBILITY ---
    # This patched function fixes two issues for CoreML conversion:
    # 1. Replaces `self.topk` with the correct attribute `self.num_top_queries`.
    # 2. Casts indices for `torch.gather` to int64 to prevent a dtype mismatch error.

    def patched_postprocessor_forward(self, outputs, orig_target_sizes):
        """A patched version of the forward pass with two critical fixes."""
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bs, nc, query = logits.shape

        prob = logits.sigmoid()
        
        # --- FIX #1: Use the correct attribute 'self.num_top_queries' instead of 'self.topk' ---
        topk_values, topk_indexes = torch.topk(prob.view(bs, -1), self.num_top_queries, dim=1)
        
        scores = topk_values
        topk_boxes = topk_indexes // self.num_classes
        labels = topk_indexes % self.num_classes
        
        # --- FIX #2: Explicitly cast the indices tensor for `torch.gather` to int64 ---
        indices = topk_boxes.unsqueeze(-1).repeat(1, 1, 4).to(torch.int64)
        boxes = torch.gather(boxes, 1, indices)
        
        # Continue with the original logic
        boxes = self.box_coder.decode(boxes)
        boxes = self.box_coder.scale(boxes, orig_target_sizes)
        
        return labels, boxes, scores

    # Replace the forward method on the *instance* of the postprocessor
    model.postprocessor.forward = types.MethodType(patched_postprocessor_forward, model.postprocessor)
    print("Applied patch to postprocessor's forward method for CoreML compatibility.")
    # --- END OF PATCH ---

    # Create an example input tensor for the tracing process.
    example_input = torch.rand(1, 3, args.input_size, args.input_size)
    print(f"Tracing model with a dummy input of shape: {example_input.shape}")
    
    try:
        traced_model = torch.jit.trace(model, example_input, strict=False)
        print("Model tracing was successful.")
    except Exception as e:
        print(f"An error occurred during model tracing: {e}")
        return

    # Define the input type for the CoreML conversion process.
    inputs = [ct.TensorType(name="images", shape=example_input.shape)]

    # Convert the traced model to the CoreML format.
    print("Starting CoreML model conversion...")
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS15,
            outputs=[
                ct.TensorType(name="labels"),
                ct.TensorType(name="boxes"),
                ct.TensorType(name="scores"),
            ],
        )

        # Add metadata to the CoreML model for better usability in Xcode.
        coreml_model.short_description = "RT-DETR model for real-time polyp detection."
        coreml_model.input_description["images"] = f"Input image, resized and normalized, of shape (1, 3, {args.input_size}, {args.input_size})."
        coreml_model.output_description["labels"] = "Predicted class label for each detection. Shape: [1, 300]."
        coreml_model.output_description["boxes"] = "Predicted bounding box coordinates (x_min, y_min, x_max, y_max) for each detection. Shape: [1, 300, 4]."
        coreml_model.output_description["scores"] = "Confidence score for each detection. Shape: [1, 300]."
        
        coreml_model.save(args.output_file)
        print(f"CoreML model conversion successful. Model saved to: {args.output_file}")

    except Exception as e:
        print(f"\nAn error occurred during CoreML conversion: {e}")

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.mlpackage')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--check', action='store_true', default=False,)
    parser.add_argument('--dynamic', action='store_true', default=False, help='Enable dynamic batch size (1-8)')

    parser = argparse.ArgumentParser(description="Export RT-DETR model to CoreML format.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to the trained PyTorch model checkpoint (.pth).')
    parser.add_argument('-o', '--output_file', type=str, default='rtdetr.mlpackage', help='Path to save the output CoreML model package.')
    parser.add_argument('-s', '--input_size', type=int, default=640, help='The input image size used for training and tracing.')
    
    args = parser.parse_args()
    main(args)