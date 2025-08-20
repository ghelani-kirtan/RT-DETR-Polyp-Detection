"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
# Add the root directory to the Python path to allow importing from 'src'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 
import coremltools as ct
import types # Import the 'types' module for monkey-patching

from src.core import YAMLConfig

def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Load the trained model weights from the checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        raise ValueError("--resume argument is required to load trained model weights.")

    # Define a wrapper class for deployment.
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images):
            outputs = self.model(images)
            orig_target_sizes = torch.tensor([[args.input_size, args.input_size]], dtype=torch.float32)
            return self.postprocessor(outputs, orig_target_sizes)

    # Instantiate the model
    model = Model()
    model.eval()

    # --- MONKEY-PATCH FOR COREML COMPATIBILITY ---
    # This patched function fixes two critical issues for CoreML conversion:
    # 1. Casts indices for `torch.gather` to int64 to prevent a dtype mismatch.
    # 2. Replaces in-place, sliced tensor modifications with out-of-place operations
    #    to prevent shape mismatch errors during conversion.

    def patched_rtdetrv2_postprocessor_forward(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        bs, nc, query = logits.shape
        logits = logits.permute(0, 2, 1)

        prob = logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.reshape(bs, -1), self.num_top_queries, dim=1)
        
        scores = topk_values
        topk_boxes = topk_indexes // logits.shape[2]
        labels = topk_indexes % logits.shape[2]

        # FIX #1: Explicitly cast the `topk_boxes` tensor to int64 for `torch.gather`.
        topk_boxes = topk_boxes.to(torch.int64)
        
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        # --- FIX #2: Convert from xywh to xyxy using out-of-place operations ---
        # This is the safe way to do this for model conversion.
        x_center, y_center, w, h = boxes.unbind(-1)
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        x_max = x_center + w / 2
        y_max = y_center + h / 2
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        return labels, boxes, scores

    # Replace the forward method on the *instance* of the postprocessor
    model.postprocessor.forward = types.MethodType(patched_rtdetrv2_postprocessor_forward, model.postprocessor)
    print("Applied final patch to rtdetrv2 postprocessor for CoreML compatibility.")
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

        coreml_model.short_description = "RT-DETR v2 model for real-time polyp detection."
        coreml_model.input_description["images"] = f"Input image, resized and normalized, of shape (1, 3, {args.input_size}, {args.input_size})."
        coreml_model.output_description["labels"] = f"Predicted class label for each of the top {model.postprocessor.num_top_queries} detections."
        coreml_model.output_description["boxes"] = "Predicted bounding box coordinates (x_min, y_min, x_max, y_max) for each detection."
        coreml_model.output_description["scores"] = "Confidence score for each detection."
        
        coreml_model.save(args.output_file)
        print(f"CoreML model conversion successful. Model saved to: {args.output_file}")

    except Exception as e:
        print(f"\nAn error occurred during CoreML conversion: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export RT-DETR v2 model to CoreML format.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to the trained PyTorch model checkpoint (.pth).')
    parser.add_argument('-o', '--output_file', type=str, default='rtdetr_v2.mlpackage', help='Path to save the output CoreML model package.')
    parser.add_argument('-s', '--input_size', type=int, default=640, help='The input image size used for training and tracing.')
    
    args = parser.parse_args()
    main(args)