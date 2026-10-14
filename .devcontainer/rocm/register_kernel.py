"""Register a venv-local notebook kernel with the AMD runtime environment."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--prefix', sys.prefix,
                '--name', 'mmdet3d-rocm', '--display-name', 'MMDetection3D (AMD ROCm)'], check=True)
kernel = Path(sys.prefix) / 'share/jupyter/kernels/mmdet3d-rocm/kernel.json'
spec = json.loads(kernel.read_text())
spec['env'] = {
    'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
    'MMDET3D_DEVICE': 'rocm',
    'PYTHONPATH': str(root),
    'MPLCONFIGDIR': str(root / '.rocm/matplotlib-cache'),
}
kernel.write_text(json.dumps(spec, indent=2) + '\n')
