"""Unpack a pinned AMD build toolchain locally, without root or apt changes."""

import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


root = Path(__file__).resolve().parents[2] / '.rocm' / 'toolchain'
root.mkdir(parents=True, exist_ok=True)
packages = json.loads(Path(__file__).with_name('toolchain-packages.json').read_text())


def download(package):
    target = root / Path(package['filename']).name
    if not target.exists():
        print(f"Downloading {package['name']}", flush=True)
        base_url = package.get('base_url', 'https://repo.radeon.com/rocm/apt/5.6/')
        urllib.request.urlretrieve(base_url + package['filename'], target)
    with target.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != package['sha256']:
        target.unlink()
        raise RuntimeError(f"Checksum mismatch: {package['name']}; rerun to download again")
    return package, target


with ThreadPoolExecutor(max_workers=4) as executor:
    # Extract sequentially: packages share directories and symlinks.
    for package, target in executor.map(download, packages):
        subprocess.run(['dpkg-deb', '--extract', str(target), str(root)], check=True)
        print(f"Extracted {package['name']}", flush=True)
