#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
cd "$project_dir"

if [[ ! -f setup.py || ! -d mmdet3d ]]; then
    echo 'The opened folder is not the MMDetection3D repository.' >&2
    exit 2
fi

python "$script_dir/link_source.py"
python "$script_dir/verify_environment.py"
python "$script_dir/verify_gpu.py"
python "$script_dir/verify_models.py"
