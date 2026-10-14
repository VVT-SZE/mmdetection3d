# AMD ROCm on this workstation

This profile targets the RX 6700 XT (12 GB, `gfx1031`) on Linux. It uses
Python 3.10, PyTorch 2.1.2 / ROCm 5.6, MMCV 2.1.0 built with HIP operators,
MMEngine 0.10.7, MMDetection 3.3.0, and this MMDetection3D 1.4 checkout.
The older stack matches the repository's existing course environment.

All AMD configuration, Docker build inputs, setup scripts, and verification
helpers live in `.devcontainer/rocm/`. Its Docker build context is this folder.
The original NVIDIA/CPU Dockerfiles, profiles, and shared helpers retain their
original contents and behavior. Both profiles use the same MMDetection3D
source checkout, with separate container environments; native AMD installs
use `.venv-rocm/` and `.rocm/`.

Verified locally on 2026-10-14: GPU GEMM/convolution, NMS/rotated NMS,
hard/dynamic voxelization, scatter backward, submanifold and strided sparse
convolutions (outputs and gradients against CPU references), PointNet
sampling/grouping/interpolation, deformable convolution, ROI Align, and
multi-scale deformable attention (outputs and gradients against CPU).
PointPillars and voxel-based CenterPoint both completed a synthetic loss,
backward pass, optimizer step, and inference on the RX 6700 XT.
Logs are in `.rocm/logs/`. The Docker image and VS Code Dev Containers CLI
build were also verified on 2026-10-14. GPU operator checks and both model
checks passed inside the container as `vscode`, using the profile's device
mappings, supplementary groups, and environment variables. Container results
are in `docker-gpu-checks.log`; VS Code's build log is `devcontainer-build.log`.

## Use the installed local environment

From the repository root, in bash or zsh:

```bash
source .devcontainer/rocm/activate.sh
python .devcontainer/rocm/verify_environment.py
python .devcontainer/rocm/verify_gpu.py
python .devcontainer/rocm/verify_models.py
jupyter lab
```

For VS Code, select `.venv-rocm/bin/python` as the Python interpreter and
the **MMDetection3D (AMD ROCm)** notebook kernel. The kernel records the
GPU compatibility override so notebooks do not depend on terminal activation.
Select the local interpreter or notebook kernel, rather than the CPU profile.

Use normal MMDetection3D commands with `cuda:0`; PyTorch deliberately exposes
AMD HIP devices through its `torch.cuda` API:

```bash
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --cfg-options train_dataloader.batch_size=1
```

Training requires preparing the dataset described by the selected configuration.
Start with batch size 1 for this card's 12 GB VRAM. CPU data loading and dataset
conversion still use the CPU. The synthetic model checks require no dataset or
checkpoint and do not measure detection accuracy.

## Recreate the native environment

```bash
bash .devcontainer/rocm/setup-native.sh
```

The bootstrap downloads Python, a compiler, and dependencies into `.rocm/`
and `.venv-rocm/`. These directories are ignored by Git. No host packages,
drivers, or system ROCm files are changed. Downloads include roughly 1.6 GB
for PyTorch and 1.3 GB for the toolchain; allow approximately 25 GB of disk
space for unpacked packages and compilation. Build concurrency defaults to 4.

The native build on Ubuntu 26.04 uses the already installed GCC 11 headers
because ROCm 5.6's Clang predates the system's GCC 15/16 headers. It also
unpacks the older linker's libxml2/ICU dependencies locally. The bootstrap
clears obsolete executable-stack flags in two old PyTorch wheel libraries
that newer glibc otherwise refuses to load.

## VS Code devcontainer

Use **Dev Containers: Reopen in Container** and select
**MMDetection3D AMD ROCm - RX 6700 XT**. A Docker runtime is required.
The image builds MMCV from source; the first build can take several minutes.
On this Linux workstation, `build.options` uses `--network=host` to avoid
Docker build DNS failures. This applies to build steps only. Apt updates
retry downloads and stop on index errors instead of continuing with missing
package lists. Installer inputs are copied separately so editing the profile
or these instructions does not trigger another MMCV compilation.
Post-create checks execute GPU NMS, voxelization, scatter, sparse convolutions,
PointNet and vision operators, with backward checks and CPU comparisons,
then synthetic training and inference for PointPillars and CenterPoint.

This machine's device group IDs are `44` (video) and `990` (render). They are
passed numerically so Ubuntu's container group names need not match the host.
On another workstation, update `runArgs` using:

```bash
stat -c '%g %n' /dev/kfd /dev/dri/renderD*
```

## Compatibility limits

The RX 6700 XT is outside AMD's validated ROCm 5.6/7.2.1 Radeon GPU list.
`HSA_OVERRIDE_GFX_VERSION=10.3.0` makes the runtime use `gfx1030` code objects;
MMCV is built for the same target. This is a compatibility workaround, not
official hardware support. Do not remove it while using these binaries.

Do not install NVIDIA MMCV wheels, `mmcv-lite`, or `spconv-cu*` into this
environment. The native build retains MMCV's legacy sparse-convolution
operators, which MMDetection3D selects automatically when spconv 2 is absent.
The operator checks are fatal on failures and never substitute CPU execution.

Passing these checks does not certify every model. Project-specific native
extensions (for example BEVFusion and DSVT), external sparse libraries, model
configurations, and mixed precision need their own HIP build and execution
validation. The NVIDIA and CPU devcontainer profiles remain available.
