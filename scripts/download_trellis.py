"""Download helper for TRELLIS / TripoSR model weights.

Usage:
    python scripts/download_trellis.py --url <download_url> [--dest tsr/model.ckpt]

It supports optional HuggingFace token via env var HF_TOKEN when downloading from private repos.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

try:
    import requests
except Exception:
    print("Missing dependency: requests. Install with `pip install requests`")
    sys.exit(1)


def download(url, dest, token=None):
    # If url points to an existing local path, skip the network download.
    p = Path(url)
    if str(url).startswith('file://'):
        p = Path(url[len('file://'):])
    if p.exists():
        print(f"[download_trellis] Local path detected, skipping download: {p}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        print(f"Downloading {url} -> {dest} ({total} bytes)")
        tmp = dest + '.part'
        with open(tmp, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        shutil.move(tmp, dest)
    print('Download complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--dest', default='tsr/model.ckpt')
    p.add_argument('--token', help='Optional access token (HF_TOKEN)')
    args = p.parse_args()
    token = args.token or os.environ.get('HF_TOKEN')
    download(args.url, args.dest, token)
