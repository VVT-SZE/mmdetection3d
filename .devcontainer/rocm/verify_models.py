"""Synthetic GPU training steps and inference for PointPillars and CenterPoint."""

import argparse
import gc

import torch

from mmdet3d.registry import MODELS
from mmdet3d.testing import create_detector_inputs, get_detector_cfg
from mmdet3d.utils import register_all_modules


CONFIGS = {
    'pointpillars': ('pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py', 4, 7, 3),
    'centerpoint': ('centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py', 5, 9, 10),
}


def verify(name):
    config, features, box_dim, classes = CONFIGS[name]
    model = MODELS.build(get_detector_cfg(config)).cuda()
    model.init_weights()
    model.train()
    packed = create_detector_inputs(num_points=4096, points_feat_dim=features,
                                    num_gt_instance=4, num_classes=classes,
                                    gt_bboxes_dim=box_dim)
    # Spread points across the actual voxel grid rather than one small corner.
    bounds = model.data_preprocessor.voxel_layer.point_cloud_range
    packed['inputs']['points'][0][:, :3] = (
        torch.rand(4096, 3) * (torch.tensor(bounds[3:]) - torch.tensor(bounds[:3]))
        + torch.tensor(bounds[:3]))
    data = model.data_preprocessor(packed, training=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    losses = model(**data, mode='loss')
    values = [value.sum() for key, loss in losses.items() if 'loss' in key
              for value in (loss if isinstance(loss, (tuple, list)) else [loss])]
    total = sum(values)
    assert torch.isfinite(total), f'{name}: non-finite loss'
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(g.is_cuda and torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        results = model(**data, mode='predict')
    assert len(results) == 1
    predictions = results[0].pred_instances_3d
    assert predictions.bboxes_3d.tensor.is_cuda
    assert torch.isfinite(predictions.bboxes_3d.tensor).all()
    torch.cuda.synchronize()
    print(f'PASS: {name}: GPU loss/backward/optimizer step and inference; loss={total.item():.4f}', flush=True)
    del model, optimizer, data, results, predictions, losses, values, total, gradients
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pointpillars', 'centerpoint', 'both'], default='both')
    args = parser.parse_args()
    assert torch.version.hip and torch.cuda.is_available(), 'A working ROCm GPU is required'
    register_all_modules()
    for name in CONFIGS if args.model == 'both' else [args.model]:
        verify(name)
