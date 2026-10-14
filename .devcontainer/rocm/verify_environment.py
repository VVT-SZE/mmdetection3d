"""Verify the AMD profile without changing NVIDIA or CPU verification."""

import mmcv
import mmdet
import mmdet3d
import mmengine
import torch
from mmcv import ops


assert torch.version.hip, 'The AMD profile requires ROCm PyTorch.'
assert torch.cuda.is_available(), 'PyTorch cannot access the AMD GPU.'
gpu_name = torch.cuda.get_device_name(0)
print(f'PyTorch {torch.__version__}, HIP {torch.version.hip}: {gpu_name}')

boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
_, keep = ops.nms(boxes, scores, 0.5)
assert keep.is_cuda and keep.tolist() == [0], 'MMCV GPU NMS returned an incorrect result.'
torch.cuda.synchronize()

print(f'MMEngine {mmengine.__version__}, MMCV {mmcv.__version__}, MMDetection {mmdet.__version__}')
print(f'MMDetection3D {mmdet3d.__version__}: {mmdet3d.__file__}')
print('AMD ROCm environment ready.')
