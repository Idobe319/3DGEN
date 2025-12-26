"""Automatic project setup helper.

Checks for TRELLIS weights and attempts to download them when a URL is
provided. Also checks for Blender and Instant Meshes and reports status.

Usage:
    python scripts/auto_setup.py

Environment variables used (optional):
- TRELLIS_URL : direct URL to model.ckpt (HF links supported with HF_TOKEN)
- HF_TOKEN    : HuggingFace token for private repos
- INSTANT_MESHES_URL : URL to Instant Meshes exe for auto-download
"""
import os
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSR_DIR = ROOT / 'tsr'
IM_PATHS = [ROOT / 'Instant Meshes.exe', ROOT / 'InstantMeshes.exe', ROOT / 'instant-meshes.exe']


def run(cmd):
    print('RUN:', ' '.join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def download_trellis(url, dest='tsr/model.ckpt', token=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if token:
        print('Using HF token from environment to download TRELLIS.')
    print(f'Downloading TRELLIS weights from {url} -> {dest}')
    cmd = [sys.executable, str(ROOT / 'scripts' / 'download_trellis.py'), '--url', url, '--dest', str(dest)]
    rc = run(cmd)
    return rc == 0


def check_blender():
    blender = shutil.which('blender')
    if blender:
        print('Blender found at', blender)
        return True
    common = [
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender\blender.exe",
    ]
    for p in common:
        if os.path.exists(p):
            print('Blender found at', p)
            return True
    print('Blender not found on PATH. Install from https://www.blender.org/')
    return False


def check_instant_meshes():
    for p in IM_PATHS:
        if p.exists():
            print('Instant Meshes found at', p)
            print('AUTO_SETUP: IM_FOUND')
            return True
    url = os.environ.get('INSTANT_MESHES_URL')
    if url:
        print('Downloading Instant Meshes from INSTANT_MESHES_URL...')
        # attempt simple download
        import requests
        target = ROOT / 'Instant Meshes.exe'
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target, 'wb') as fh:
                for chunk in r.iter_content(8192):
                    if chunk:
                        fh.write(chunk)
        if target.exists():
            print('Downloaded Instant Meshes ->', target)
            print('AUTO_SETUP: IM_FOUND')
            return True
    print('Instant Meshes executable not found. You can set INSTANT_MESHES_URL to auto-download one.')
    print('AUTO_SETUP: IM_NOT_FOUND')
    return False


def main():
    print('Auto-setup: Starting checks...')
    trellis_url = os.environ.get('TRELLIS_URL')
    trellis_repo = os.environ.get('TRELLIS_HF_REPO', 'stabilityai/TripoSR')

    cfg_path = TSR_DIR / 'config.yaml'
    ckpt_path = TSR_DIR / 'model.ckpt'

    # Ensure config.yaml is present (needed for TripoSR to run offline)
    if not cfg_path.exists():
        print('config.yaml missing; attempting to download from HF repo', trellis_repo)
        rc = run([sys.executable, str(ROOT / 'scripts' / 'hf_download_trellis.py'), '--repo', trellis_repo, '--config-dest', str(cfg_path), '--only-config'])
        if rc == 0:
            print('Downloaded config.yaml ->', cfg_path)
            print('AUTO_SETUP: CONFIG_OK')
        else:
            print('Failed to download config.yaml from HF repo', trellis_repo)
            print('AUTO_SETUP: CONFIG_FAIL')

    # Ensure weights are present; prefer TRELLIS_URL if specified, otherwise try HF repo
    if not ckpt_path.exists():
        if trellis_url:
            ok = download_trellis(trellis_url, dest=str(ckpt_path), token=os.environ.get('HF_TOKEN'))
            print('TRELLIS download ok:', ok)
            print('AUTO_SETUP: WEIGHTS_OK' if ok else 'AUTO_SETUP: WEIGHTS_FAIL')
        else:
            print('No TRELLIS_URL supplied; attempting to download candidate weights from HF repo', trellis_repo)
            rc = run([sys.executable, str(ROOT / 'scripts' / 'hf_download_trellis.py'), '--repo', trellis_repo, '--dest', str(ckpt_path)])
            if rc == 0:
                print('Downloaded candidate weights ->', ckpt_path)
                print('AUTO_SETUP: WEIGHTS_OK')
            else:
                print('No weights downloaded from HF repo (this may be normal for private/protected models).')
                print('AUTO_SETUP: WEIGHTS_NONE')
    else:
        print('Found local weights at', ckpt_path)
        print('AUTO_SETUP: WEIGHTS_PRESENT')

    check_blender()
    check_instant_meshes()
    print('Auto-setup: done.')
    print('AUTO_SETUP: DONE')


if __name__ == '__main__':
    main()
