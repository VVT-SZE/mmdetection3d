"""Execute representative MMDetection3D GPU kernels and their backward paths.

Run after activating the ROCm environment. Failures are fatal, never skipped.
This validates shared operators, not every model or project-specific extension.
"""

import torch
from mmcv import ops
from torch.nn import functional as F


def check(name, function):
    function()
    torch.cuda.synchronize()
    print(f'PASS: {name}', flush=True)


def check_torch():
    x = torch.randn(128, 128, device='cuda', requires_grad=True)
    result = x @ x
    torch.testing.assert_close(result.cpu(), x.detach().cpu() @ x.detach().cpu())
    result.square().mean().backward()
    assert x.grad.is_cuda and torch.isfinite(x.grad).all()
    layer = torch.nn.Conv2d(4, 8, 3).cuda()
    layer(torch.randn(2, 4, 16, 16, device='cuda')).square().mean().backward()
    assert layer.weight.grad.is_cuda and torch.isfinite(layer.weight.grad).all()


def check_nms():
    scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.],
                          [5., 5., 7., 7.]], device='cuda')
    _, keep = ops.nms(boxes, scores, 0.5)
    assert keep.is_cuda and keep.tolist() == [0, 2]
    rotated = torch.tensor([[1., 1., 2., 2., 0.], [1., 1., 2., 2., 0.],
                            [6., 6., 2., 2., 0.]], device='cuda')
    _, keep = ops.nms_rotated(rotated, scores, 0.5)
    assert keep.is_cuda and keep.tolist() == [0, 2]


def check_voxels():
    from mmdet3d.models.data_preprocessors.voxelize import (
        dynamic_scatter_3d, voxelization)

    points = torch.tensor([[.1, .1, .1, 1.], [.2, .2, .2, 3.],
                           [1.1, 1.1, 1.1, 5.]])
    args = ([1., 1., 1.], [0., 0., 0., 4., 4., 4.])
    cpu = voxelization(points, *args, 4, 16, True)
    gpu = voxelization(points.cuda(), *args, 4, 16, True)
    for expected, actual in zip(cpu, gpu):
        assert actual.is_cuda
        torch.testing.assert_close(actual.cpu(), expected)
    coors = voxelization(points.cuda(), *args, -1, -1, True)
    torch.testing.assert_close(coors.cpu(), torch.tensor([[0, 0, 0],
                                                        [0, 0, 0],
                                                        [1, 1, 1]], dtype=torch.int))
    features = points[:, 3:].cuda().requires_grad_()
    reduced, _ = dynamic_scatter_3d(features, coors, 'mean')
    torch.testing.assert_close(reduced.cpu(), torch.tensor([[2.], [5.]]))
    reduced.sum().backward()
    torch.testing.assert_close(features.grad.cpu(), torch.tensor([[.5], [.5], [1.]]))


def check_sparse():
    # Exercise the exact fallback MMDetection3D uses when spconv 2 is absent.
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

    assert not IS_SPCONV2_AVAILABLE, 'ROCm must use MMCV sparse ops, not CUDA spconv 2'
    xyz = torch.cartesian_prod(*(torch.arange(2, 6) for _ in range(3)))
    indices = torch.cat((torch.zeros(len(xyz), 1, dtype=torch.int), xyz.int()), 1)
    for cls, stride in [(ops.SubMConv3d, 1), (ops.SparseConv3d, 2)]:
        features = torch.randn(len(xyz), 2, device='cuda', requires_grad=True)
        layer = cls(2, 4, 3, stride=stride, padding=1, bias=False).cuda()
        sparse = ops.SparseConvTensor(features, indices.cuda(), [8, 8, 8], 1)
        result = layer(sparse)
        assert result.features.is_cuda
        dense = sparse.dense().detach().cpu().requires_grad_()
        weight = layer.weight.detach().cpu().permute(4, 3, 0, 1, 2).contiguous()
        weight.requires_grad_()
        expected = F.conv3d(dense, weight, stride=stride, padding=1)
        coords = result.indices.cpu().long()
        expected_features = expected[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]
        torch.testing.assert_close(result.features.cpu(), expected_features, rtol=1e-4, atol=1e-5)
        result.features.square().mean().backward()
        expected_features.square().mean().backward()
        torch.testing.assert_close(layer.weight.grad.cpu().permute(4, 3, 0, 1, 2),
                                   weight.grad, rtol=1e-4, atol=1e-5)
        expected_grad = dense.grad[0, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]].T
        torch.testing.assert_close(features.grad.cpu(), expected_grad, rtol=1e-4, atol=1e-5)


def check_pointnet():
    xyz = torch.rand(1, 32, 3, device='cuda')
    sampled = ops.furthest_point_sample(xyz, 8)
    assert sampled.is_cuda and sampled.unique().numel() == 8
    queries = xyz[:, :8].contiguous()
    indices = ops.ball_query(0., 2., 8, xyz, queries)
    features = torch.rand(1, 4, 32, device='cuda', requires_grad=True)
    grouped = ops.grouping_operation(features, indices)
    grouped.sum().backward()
    assert features.grad.is_cuda and torch.isfinite(features.grad).all()
    distances, neighbors = ops.three_nn(queries, xyz)
    # Avoid cdist's GEMM shortcut: cancellation makes self-distances nonzero.
    expected_distances = torch.cdist(queries.cpu(), xyz.cpu(),
                                    compute_mode='donot_use_mm_for_euclid_dist')
    torch.testing.assert_close(distances.cpu(), expected_distances.topk(3, largest=False).values,
                               rtol=1e-4, atol=1e-4)
    features.grad = None
    weights = torch.full_like(distances, 1 / 3)
    interpolated = ops.three_interpolate(features, neighbors, weights)
    interpolated.sum().backward()
    assert features.grad.is_cuda and torch.isfinite(features.grad).all()


def check_vision():
    inputs = torch.randn(2, 2, 8, 8, device='cuda', requires_grad=True)
    offsets = torch.zeros(2, 18, 8, 8, device='cuda', requires_grad=True)
    weight = torch.randn(4, 2, 3, 3, device='cuda', requires_grad=True)
    result = ops.deform_conv2d(inputs, offsets, weight, 1, 1)
    torch.testing.assert_close(result.cpu(), F.conv2d(inputs.detach().cpu(), weight.detach().cpu(), padding=1),
                               rtol=1e-4, atol=1e-5)
    result.square().mean().backward()
    assert all(x.grad.is_cuda and torch.isfinite(x.grad).all() for x in (inputs, offsets, weight))
    grid = torch.arange(64, device='cuda', dtype=torch.float).reshape(1, 1, 8, 8).requires_grad_()
    rois = torch.tensor([[0., 1., 1., 5., 5.]], device='cuda')
    pooled = ops.roi_align(grid, rois, (2, 2), 1., 2, 'avg', True)
    torch.testing.assert_close(pooled.cpu(), torch.tensor([[[[13.5, 15.5], [29.5, 31.5]]]]))
    pooled.sum().backward()
    assert grid.grad.is_cuda and torch.isfinite(grid.grad).all()


def check_attention():
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)

    value = torch.randn(2, 4, 2, 2, device='cuda', requires_grad=True)
    shapes = torch.tensor([[2, 2]], device='cuda', dtype=torch.long)
    starts = torch.zeros(1, device='cuda', dtype=torch.long)
    locations = torch.rand(2, 2, 2, 1, 2, 2, device='cuda', requires_grad=True)
    weights = torch.full((2, 2, 2, 1, 2), .5, device='cuda', requires_grad=True)
    result = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 2)
    cpu_inputs = [x.detach().cpu().requires_grad_() for x in (value, locations, weights)]
    expected = multi_scale_deformable_attn_pytorch(cpu_inputs[0], shapes.cpu(), cpu_inputs[1], cpu_inputs[2])
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)
    result.square().mean().backward()
    expected.square().mean().backward()
    for gpu, cpu in zip((value, locations, weights), cpu_inputs):
        assert gpu.grad.is_cuda
        torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    assert torch.version.hip and torch.cuda.is_available(), 'A working ROCm PyTorch GPU is required'
    torch.manual_seed(0)
    print(f'{torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}; HIP {torch.version.hip}', flush=True)
    check('PyTorch GEMM and convolution forward/backward', check_torch)
    check('NMS and rotated NMS', check_nms)
    check('hard/dynamic voxelization and scatter backward', check_voxels)
    check('submanifold/strided sparse convolution vs CPU, including backward', check_sparse)
    check('PointNet sampling/grouping/interpolation and backward', check_pointnet)
    check('deformable convolution and ROI Align forward/backward', check_vision)
    check('multi-scale deformable attention vs CPU, including backward', check_attention)
