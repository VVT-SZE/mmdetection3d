#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
cd "$project_dir"
mkdir -p .rocm/tools .rocm/logs
command -v dpkg-deb >/dev/null || { echo 'This bootstrap requires Debian/Ubuntu dpkg-deb.' >&2; exit 1; }
if [ ! -x .rocm/tools/uv-x86_64-unknown-linux-gnu/uv ]; then
    curl --fail --location --retry 3 \
        https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz \
        -o .rocm/tools/uv.tar.gz
    printf '%s  %s\n' 'eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220' \
        .rocm/tools/uv.tar.gz | sha256sum --check
    tar -xzf .rocm/tools/uv.tar.gz -C .rocm/tools
fi
export UV_CACHE_DIR="$project_dir/.rocm/cache"
export UV_PYTHON_INSTALL_DIR="$project_dir/.rocm/python"
if [ ! -x .venv-rocm/bin/python ]; then
    .rocm/tools/uv-x86_64-unknown-linux-gnu/uv venv --seed --python 3.10 .venv-rocm
fi
source "$script_dir/activate.sh"
python "$script_dir/fetch_toolchain.py"

# ROCm's Clang 16 predates Ubuntu 26.04's GCC 15/16 headers. Use the
# installed GCC 11 headers for the host and device compilations.
if [ -d /usr/lib/gcc/x86_64-linux-gnu/11 ] && [ -d /usr/include/c++/11 ]; then
    cat > .rocm/toolchain/cxx <<'SH'
#!/usr/bin/env bash
exec "$(dirname "$0")/opt/rocm-5.6.0/llvm/bin/clang++" --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 "$@"
SH
    chmod +x .rocm/toolchain/cxx
    export CXX="$project_dir/.rocm/toolchain/cxx"
    export HIPCC_COMPILE_FLAGS_APPEND=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
fi
export MMDET3D_MMCV_SOURCE="$project_dir/.rocm/sources"
bash "$script_dir/install-stack.sh" 2>&1 | tee .rocm/logs/install-stack.log
python "$script_dir/link_source.py"
python "$script_dir/verify_environment.py"
python "$script_dir/verify_gpu.py" 2>&1 | tee .rocm/logs/verify-gpu.log
python "$script_dir/verify_models.py" 2>&1 | tee .rocm/logs/verify-models.log
python "$script_dir/register_kernel.py"
