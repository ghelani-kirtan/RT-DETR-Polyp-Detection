""" "Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import torch
import torch.nn as nn

import torchvision

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image
import cv2
import numpy as np

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register

RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomEqualize = register()(T.RandomEqualize)
ColorJitter = register()(T.ColorJitter)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(
                inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower()
            )
            inpt = convert_to_tv_tensor(
                inpt,
                key="boxes",
                box_format=self.fmt.upper(),
                spatial_size=spatial_size,
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class CLAHEEnhance(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
        Image,
    )

    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=1.0):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # (Your existing CLAHE logic is here - no changes needed)
        if isinstance(inpt, PIL.Image.Image):
            image_np = np.array(inpt)
        elif isinstance(inpt, Image):
            image_np = inpt.permute(1, 2, 0).numpy()
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = inpt.permute(1, 2, 0).numpy().astype(np.uint8)

        if image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        image_lab[:, :, 0] = clahe.apply(image_lab[:, :, 0])
        enhanced_np = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

        if isinstance(inpt, PIL.Image.Image):
            return PIL.Image.fromarray(enhanced_np)
        elif isinstance(inpt, Image):
            enhanced_tensor = torch.from_numpy(enhanced_np).permute(2, 0, 1)
            return Image(enhanced_tensor)

        return torch.from_numpy(enhanced_np).permute(2, 0, 1)

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)

    # --- ADD THIS METHOD ---
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """This method is called by the base class's forward()"""
        return self._transform(inpt, params)


@register()
class NBISimulation(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
        Image,
    )

    def __init__(self, blue_factor=1.5, green_factor=1.2, red_factor=0.8, p=1.0):
        super().__init__()
        self.blue_factor = blue_factor
        self.green_factor = green_factor
        self.red_factor = red_factor
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # Convert to NumPy array for channel manipulation
        if isinstance(inpt, PIL.Image.Image):
            image_np = np.array(inpt).astype(np.float32)
        else:  # Handles custom Image class or general tensors
            # Assuming tensor is (C, H, W), convert to (H, W, C) numpy
            image_np = inpt.permute(1, 2, 0).numpy().astype(np.float32)

        # Apply NBI simulation logic
        image_np[:, :, 2] *= self.blue_factor
        image_np[:, :, 1] *= self.green_factor
        image_np[:, :, 0] *= self.red_factor

        # Clip values to valid range and convert back to uint8
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # Convert back to the original type
        if isinstance(inpt, PIL.Image.Image):
            return PIL.Image.fromarray(image_np)
        else:
            # Convert back to tensor (C, H, W) and wrap if needed
            enhanced_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            return (
                Image(enhanced_tensor) if isinstance(inpt, Image) else enhanced_tensor
            )

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class RandomColorJitter(T.RandomApply):
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        # 1. Create the core ColorJitter transform from the arguments
        jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

        # 2. Call the parent constructor (T.RandomApply) to wrap it
        #    The first argument must be a list of transforms.
        super().__init__([jitter], p=p)
