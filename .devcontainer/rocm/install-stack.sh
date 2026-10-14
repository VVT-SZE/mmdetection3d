#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for this pinned stack"'
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
export PYTORCH_ROCM_ARCH=gfx1030
export MAX_JOBS="${MAX_JOBS:-4}"
export CFLAGS="${CFLAGS:--g0 -O2}"
export CXXFLAGS="${CXXFLAGS:--g0 -O2}"
python -m pip install 'pip<25' 'setuptools==69.5.1' wheel ninja
python -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/rocm5.6
python "$script_dir/clear_execstack.py"
python -m pip install -r "$script_dir/requirements.txt"

# Keep CUDA/CPU wheels and spconv 2 out of the AMD environment.
python -m pip uninstall -y mmcv mmcv-lite
source_dir="${MMDET3D_MMCV_SOURCE:-$VIRTUAL_ENV/mmcv-source}"
mkdir -p "$source_dir"
curl --fail --location --retry 3 \
    https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.1.0.tar.gz \
    -o "$source_dir/mmcv.tar.gz"
printf '%s  %s\n' '4e9d28b42179404aff9ce4749e2cc5a3ad45fe76e3a0621608bcc8e144e917a1' \
    "$source_dir/mmcv.tar.gz" | sha256sum --check
tar -xzf "$source_dir/mmcv.tar.gz" -C "$source_dir"
(
    cd "$source_dir/mmcv-2.1.0"
    # Preserve intermediate objects on failure so a retry can resume.
    MMCV_WITH_OPS=1 python setup.py bdist_wheel
    python -m pip install dist/mmcv-2.1.0-*.whl
)
python -m pip check
