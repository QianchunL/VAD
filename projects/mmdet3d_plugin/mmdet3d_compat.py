"""Compatibility shims for mmdet3d 0.17.x -> 1.0.x API changes.

In mmdet3d 1.0+, most data structures moved from mmdet3d.core to
mmdet3d.structures. This module provides fallback imports so VAD
(originally written for mmdet3d 0.17.1) works on 1.0.0a1+.
"""

# --- ops ---

try:
    from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
except ImportError:
    from mmdet3d.ops import points_in_boxes_gpu

# --- bbox3d2result ---

try:
    from mmdet3d.core import bbox3d2result
except (ImportError, AttributeError):
    try:
        from mmdet3d.models.utils import bbox3d2result
    except ImportError:
        from mmdet3d.utils import bbox3d2result

# --- MVXTwoStageDetector ---

try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
except ImportError:
    from mmdet3d.models import MVXTwoStageDetector

# --- 3D box structures ---

try:
    from mmdet3d.core import LiDARInstance3DBoxes
except (ImportError, AttributeError):
    from mmdet3d.structures import LiDARInstance3DBoxes

try:
    from mmdet3d.core.bbox.structures.base_box3d import BaseInstance3DBoxes
except ImportError:
    from mmdet3d.structures import BaseInstance3DBoxes

try:
    from mmdet3d.core.bbox.structures.utils import limit_period, rotation_3d_in_axis
except ImportError:
    from mmdet3d.structures import limit_period, rotation_3d_in_axis

try:
    from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                   LiDARInstance3DBoxes, box_np_ops)
except ImportError:
    from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                    LiDARInstance3DBoxes)
    try:
        from mmdet3d.ops import box_np_ops
    except ImportError:
        from mmdet3d.structures import box_np_ops

try:
    from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
except ImportError:
    from mmdet3d.structures import Box3DMode

try:
    from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D
except ImportError:
    from mmdet3d.structures import BboxOverlaps3D

# --- coders ---

try:
    from mmdet3d.core.bbox.coders import build_bbox_coder
except ImportError:
    try:
        from mmdet3d.models.task_modules.coders import build_bbox_coder
    except ImportError:
        from mmdet3d.models.task_modules import build_bbox_coder

# --- points ---

try:
    from mmdet3d.core.points import BasePoints, get_points_type
except ImportError:
    from mmdet3d.structures import BasePoints, get_points_type

# --- datasets ---

try:
    from mmdet3d.datasets import NuScenesDataset
except ImportError:
    from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

try:
    from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
except ImportError:
    try:
        from mmdet3d.datasets.transforms import DefaultFormatBundle3D
    except ImportError:
        from mmdet3d.datasets import DefaultFormatBundle3D

try:
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
except ImportError:
    from mmdet3d.datasets import CBGSDataset
