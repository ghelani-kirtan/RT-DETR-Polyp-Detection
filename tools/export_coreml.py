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
import torch.nn.functional as F

from src.core import YAMLConfig

def patched_ms_deform_attn_forward(self, query, reference_points, value, value_spatial_shapes, value_level_start_index, value_mask=None):
    bs, Len_q = query.shape[:2]
    Len_v = value.shape[1]

    value = self.value_proj(value)
    if value_mask is not None:
        value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
        value *= value_mask
    embed_dim = value.shape[-1]
    value = value.view(bs, Len_v, self.num_heads, self.head_dim)

    merged_q = Len_q * self.num_heads

    # sampling_offsets (bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2) -> (bs, merged_q, self.num_levels, self.num_points, 2)
    sampling_offsets = self.sampling_offsets(query).view(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2).flatten(1, 2)

    # attention_weights (bs, Len_q, self.num_heads, self.num_levels*self.num_points) -> softmax -> (bs, merged_q, self.num_levels, self.num_points)
    attention_weights = self.attention_weights(query).view(bs, Len_q, self.num_heads, self.num_levels * self.num_points).flatten(1, 2)
    attention_weights = F.softmax(attention_weights, -1).view(bs, merged_q, self.num_levels, self.num_points)

    # reference_points (bs, Len_q, self.num_levels, ref_dim) -> (bs, merged_q, self.num_levels, ref_dim)
    ref_dim = reference_points.shape[-1]
    reference_points = reference_points.repeat_interleave(self.num_heads, 1)

    if ref_dim == 2:
        offset_normalizer = value_spatial_shapes.flip(1)  # (self.num_levels, 2) [w, h]
        sampling_locations = reference_points.unsqueeze(2) + sampling_offsets / offset_normalizer[None, None, None, :, None]
    elif ref_dim == 4:
        sampling_locations = reference_points[:, :, None, :, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, 2:] * 0.5
    else:
        raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {ref_dim}')

    output = torch.zeros(bs * self.num_heads, embed_dim, dtype=value.dtype, device=value.device)

    for lvl in range(self.num_levels):
        level_start = value_level_start_index[lvl]
        h, w = value_spatial_shapes[lvl]
        level_end = level_start + int(h * w)

        value_l = value[:, level_start:level_end, :, :].view(bs, h, w, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)  # (bs, num_heads, head_dim, h, w)
        value_l = value_l.flatten(0, 1)  # (bs * num_heads, head_dim, h, w)

        sampling_locations_l = sampling_locations[:, :, lvl, :, :]  # (bs, merged_q, self.num_points, 2)

        # Normalize to -1 to 1
        sampling_grid_l = 2 * sampling_locations_l - 1
        sampling_grid_l = sampling_grid_l.flip(-1).view(bs, merged_q, self.num_points, 2)  # (bs, merged_q, p, 2) with (y, x)

        # To make grid_sample work with 1D output, use H_out=1, W_out=p
        sampled = F.grid_sample(value_l.repeat_interleave(merged_q // self.num_heads, dim=0), sampling_grid_l.view(bs * merged_q, 1, self.num_points, 2),
                                mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(2)  # (bs * merged_q, head_dim, p)

        attn_l = attention_weights[:, :, lvl, :].view(bs * merged_q, 1, self.num_points)  # (bs * merged_q, 1, p)

        weighted = sampled * attn_l
        out_l = weighted.sum(-1)  # (bs * merged_q, head_dim)

        output += out_l

    # Reshape back
    output = output.view(bs, merged_q, self.head_dim)
    output = output.view(bs, Len_q, self.num_heads, self.head_dim).permute(0, 1, 2, 3).contiguous().view(bs, Len_q, embed_dim)

    output = self.output_proj(output)

    return output

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

    # Apply patches to all cross_attn in decoder layers
    for layer in model.model.decoder.layers:
        layer.cross_attn.forward = types.MethodType(patched_ms_deform_attn_forward, layer.cross_attn)
    print("Applied patch to MSDeformableAttention forward for CoreML compatibility (rank <=5).")

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