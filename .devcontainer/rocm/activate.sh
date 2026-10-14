# Source this file from bash or zsh; it does not change the host ROCm install.
if [ -n "${ZSH_VERSION:-}" ]; then
    _mmdet_rocm_script="${(%):-%x}"
else
    _mmdet_rocm_script="${BASH_SOURCE[0]}"
fi
_mmdet_rocm_root="$(cd "$(dirname "$_mmdet_rocm_script")/../.." && pwd)"
source "$_mmdet_rocm_root/.venv-rocm/bin/activate"
export MMDET3D_DEVICE=rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export ROCM_HOME="$_mmdet_rocm_root/.rocm/toolchain/opt/rocm-5.6.0"
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HIP_CLANG_PATH="$ROCM_HOME/llvm/bin"
export PATH="$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$_mmdet_rocm_root/.rocm/toolchain/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x "$_mmdet_rocm_root/.rocm/toolchain/cxx" ]; then
    export CXX="$_mmdet_rocm_root/.rocm/toolchain/cxx"
    export HIPCC_COMPILE_FLAGS_APPEND=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
fi
export PYTHONPATH="$_mmdet_rocm_root${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$_mmdet_rocm_root/.rocm/matplotlib-cache"
unset _mmdet_rocm_script _mmdet_rocm_root
