"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

import numpy as np

from src.core import YAMLConfig

import coremltools as ct
from coremltools import ComputeUnit, TensorType, RangeDim

def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)
    else:
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
           
        def forward(self, images, orig_target_sizes):
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
    ct_model.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(params)

    # Save the model
    ct_model.save(args.output_file)
    print(f'Exported CoreML model to {args.output_file}')

    if args.check:
        # Test prediction with random input
        test_images = np.random.rand(1, 3, args.input_size, args.input_size).astype(np.float32)
        test_sizes = np.array([[args.input_size, args.input_size]], dtype=np.float32)
        predictions = ct_model.predict({'images': test_images, 'orig_target_sizes': test_sizes})
        print('Check export CoreML model done...')
        print('Sample output shapes:')
        print(f"labels: {predictions['labels'].shape}")
        print(f"boxes: {predictions['boxes'].shape}")
        print(f"scores: {predictions['scores'].shape}")

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

    args = parser.parse_args()
    main(args)