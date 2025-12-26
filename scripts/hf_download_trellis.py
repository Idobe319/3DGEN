"""Download TRELLIS/TripoSR weights (and config) from Hugging Face.

Usage:
    python scripts/hf_download_trellis.py --repo stabilityai/TripoSR \
        --dest tsr/model.ckpt --config-dest tsr/config.yaml

This will download a candidate weights file and also fetch `config.yaml`
so TripoSR can run offline with local files. If `HF_TOKEN` is set it will be used.
"""
import argparse
import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

CANDIDATE_NAMES = [
    'model.ckpt',
    'pytorch_model.bin',
    'pytorch_model.bin.index.json',
    'pytorch_model-00001-of-00002.bin',
    'model.safetensors',
]


def find_candidate(repo_id):
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print('Failed to list repo files:', e)
        files = []
    for name in CANDIDATE_NAMES:
        if name in files:
            return name
    # fallback: try to find any file with .ckpt or .safetensors
    for f in files:
        if f.lower().endswith('.ckpt') or f.lower().endswith('.safetensors') or f.lower().endswith('.bin'):
            return f
    return None


def download_file(repo_id: str, filename: str, dest: str, token: str | None = None) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        path = hf_hub_download(repo_id, filename=filename, repo_type='model', token=token)
        shutil.copy(path, dest)
        print(f'Downloaded {filename} -> {dest}')
        return True
    except Exception as e:
        print(f'Failed to download {filename}:', e)
        return False


def download_weights(repo_id: str, dest: str, token: str | None = None) -> bool:
    candidate = find_candidate(repo_id)
    if not candidate:
        print('No candidate weight files found in repo', repo_id)
        return False
    print('Found candidate weight file:', candidate)
    return download_file(repo_id, candidate, dest, token)


def download_config(repo_id: str, dest: str, token: str | None = None) -> bool:
    # Prefer `config.yaml` in the repo root.
    print('Attempting to download config.yaml from', repo_id)
    return download_file(repo_id, 'config.yaml', dest, token)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='stabilityai/TripoSR')
    p.add_argument('--dest', default='tsr/model.ckpt', help='Destination path for model weights')
    p.add_argument('--config-dest', default='tsr/config.yaml', help='Destination for TripoSR config.yaml')
    p.add_argument('--token', help='HF_TOKEN (optional)')
    p.add_argument('--only-config', action='store_true', help='Only download config.yaml')
    args = p.parse_args()
    tok = args.token or os.environ.get('HF_TOKEN')

    if args.only_config:
        ok_cfg = download_config(args.repo, args.config_dest, tok)
        if not ok_cfg:
            raise SystemExit(1)
        print('Config download complete.')
        raise SystemExit(0)

    ok_weights = download_weights(args.repo, args.dest, tok)
    ok_cfg = download_config(args.repo, args.config_dest, tok)

    if not ok_weights:
        print('Warning: failed to download weights (but config status =', ok_cfg, ').')
        raise SystemExit(1)

    print('Download finished. Weights: {}, Config: {}'.format(ok_weights, ok_cfg))
