"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

import numpy as np
import onnx

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

    onnx_path = args.output_file.replace('.mlpackage', '.onnx') if args.output_file.endswith('.mlpackage') else 'temp.onnx'

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    data = torch.rand(1, 3, args.input_size, args.input_size)
    size = torch.tensor([[args.input_size, args.input_size]], dtype=torch.float32)

    torch.onnx.export(
        model,
        (data, size),
        onnx_path,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.simplify:
        import onnxsim
        onnx_model = onnx.load(onnx_path)
        dynamic = args.dynamic
        input_shapes = {'images': list(data.shape), 'orig_target_sizes': list(size.shape)} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, onnx_path)
        print(f'Simplify onnx model {check}...')

    # Load ONNX model and insert casts for Gather indices
    onnx_model = onnx.load(onnx_path)
    new_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == 'Gather':
            indices_input_name = node.input[1]
            cast_output_name = indices_input_name + '_cast_int32'
            cast_node = onnx.helper.make_node(
                'Cast',
                inputs=[indices_input_name],
                outputs=[cast_output_name],
                to=onnx.TensorProto.INT32,
            )
            new_nodes.append(cast_node)
            node.input[1] = cast_output_name
        new_nodes.append(node)
    onnx_model.graph.node[:] = new_nodes

    if args.check:
        import onnx
        onnx.checker.check_model(onnx_model)
        print('Check modified onnx model done...')

    # Define input shapes for CoreML
    batch_dim = RangeDim(lower_bound=1, upper_bound=1, default=1) if not args.dynamic else RangeDim(lower_bound=1, upper_bound=8, default=1)
    images_shape = (batch_dim, 3, args.input_size, args.input_size)
    sizes_shape = (batch_dim, 2)

    images_input = TensorType(name='images', shape=images_shape, dtype=np.float32)
    sizes_input = TensorType(name='orig_target_sizes', shape=sizes_shape, dtype=np.float32)

    # Convert to CoreML
    ct_model = ct.convert(
        onnx_model,
        convert_to='mlprogram',
        inputs=[images_input, sizes_input],
        outputs=[
            ct.TensorType(name='labels', dtype=np.int32),
            ct.TensorType(name='boxes', dtype=np.float32),
            ct.TensorType(name='scores', dtype=np.float32)
        ],
        compute_units=ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )

    # Optional: Set metadata for object detector preview in Xcode
    labels = ['background', 'polyp']  # Adjust based on classes
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
    parser.add_argument('--simplify', action='store_true', default=False,)
    parser.add_argument('--dynamic', action='store_true', default=False, help='Enable dynamic batch size (1-8)')

    args = parser.parse_args()
    main(args)