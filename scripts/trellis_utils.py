import os
import json
import glob
from typing import Set, List, Optional

_EXTENSIONS = ('.ckpt', '.pt', '.pth', '.safetensors', '.safetensor')


def _strings_from_obj(obj):
    """Recursively yields string values from nested dict/list structures."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings_from_obj(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings_from_obj(v)
    elif isinstance(obj, str):
        yield obj


def discover_expected_ckpts(project_root: Optional[str] = None) -> Set[str]:
    """Scan TRELLIS config files to infer expected checkpoint filenames.

    Returns a set of filenames (basenames) commonly found in config values
    that reference `ckpts/` or end with a known checkpoint extension.
    If `project_root` is None the current working directory is used.
    """
    pr = project_root or os.getcwd()
    cfg_dir = os.path.join(pr, 'external', 'TRELLIS', 'configs')
    expected = set()

    if not os.path.isdir(cfg_dir):
        return expected

    for cfg_path in glob.glob(os.path.join(cfg_dir, '**', '*.json'), recursive=True):
        try:
            with open(cfg_path, 'r', encoding='utf8') as fh:
                j = json.load(fh)
        except Exception:
            # ignore parse errors on non-json files
            continue
        for s in _strings_from_obj(j):
            if 'ckpts' in s:
                # extract basename (strip extension if present) so we can try multiple suffixes later
                base = os.path.basename(s)
                if base:
                    stem, ext = os.path.splitext(base)
                    if ext and ext.lower() in _EXTENSIONS:
                        expected.add(stem)
                    else:
                        expected.add(base if not ext else stem)
            else:
                # standalone filename references
                base = os.path.basename(s)
                if base:
                    stem, ext = os.path.splitext(base)
                    if ext and ext.lower() in _EXTENSIONS:
                        expected.add(stem)
    return expected


def check_model_dir_for_ckpts(model_dir: str, expected: Optional[Set[str]] = None) -> List[str]:
    """Return a list of missing checkpoint filenames in model_dir/ckpts given expected set.

    `expected` should be a set of checkpoint basenames (no extension). For each expected
    base we consider a small set of candidate extensions and deem the checkpoint present
    if any candidate file exists.
    """
    if expected is None:
        expected = discover_expected_ckpts(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    missing = []
    ckpt_dir = os.path.join(model_dir, 'ckpts')
    for stem in sorted(expected):
        # if an explicit file exists with any known extension, consider it present
        found = False
        for ext in _EXTENSIONS:
            if os.path.exists(os.path.join(ckpt_dir, stem + ext)):
                found = True
                break
        if not found:
            # report preferred filename (.safetensors first) followed by other candidates
            preferred = stem + '.safetensors'
            candidates = [stem + e for e in _EXTENSIONS]
            missing.extend(candidates)
    return missing


def discover_ckpt_sources(project_root: Optional[str] = None) -> dict:
    """Try to infer a mapping from checkpoint basename -> HF repo id or URL.

    It scans TRELLIS configs for strings that contain repository identifiers
    (e.g., 'microsoft/TRELLIS-image-large') and checkpoint filenames.
    Returns a dict filename -> repo_id (strings).
    """
    pr = project_root or os.getcwd()
    cfg_dir = os.path.join(pr, 'external', 'TRELLIS', 'configs')
    mapping = {}
    if not os.path.isdir(cfg_dir):
        return mapping

    for cfg_path in glob.glob(os.path.join(cfg_dir, '**', '*.json'), recursive=True):
        try:
            with open(cfg_path, 'r', encoding='utf8') as fh:
                j = json.load(fh)
        except Exception:
            continue
        for s in _strings_from_obj(j):
            if not isinstance(s, str):
                continue
            # look for patterns like 'microsoft/TRELLIS-image-large/ckpts/filename'
            if 'microsoft/TRELLIS-image-large' in s:
                # extract filename if present
                base = os.path.basename(s)
                if base and base.lower().endswith(_EXTENSIONS):
                    mapping[base] = 'microsoft/TRELLIS-image-large'
    return mapping


def ckpt_hint(fn: str, mapping: Optional[dict] = None) -> str:
    """Return a human-friendly Hugging Face hint (repo and raw URL) for a given ckpt basename."""
    if mapping is None:
        mapping = discover_ckpt_sources()
    repo = mapping.get(fn)
    if not repo:
        repo = 'microsoft/TRELLIS-image-large'

    # If offline mode is enabled, prefer local-path hints instead of HF URLs
    if os.environ.get('HF_HUB_OFFLINE') or os.environ.get('TRANSFORMERS_OFFLINE'):
        pr = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        local1 = os.path.join(pr, 'tsr', 'ckpts', fn)
        local2 = os.path.join(pr, 'tsr', 'TRELLIS-image-large', 'ckpts', fn)
        return f'Put the checkpoint file at one of these local paths: {local1} or {local2}'

    # prefer resolved path; Hugging Face raw access via /resolve/main/ckpts/<fn>
    hf_url = f'https://huggingface.co/{repo}/resolve/main/ckpts/{fn}'
    return f'{repo}/ckpts/{fn} ({hf_url})'


def download_missing_ckpts(model_dir: str, expected: Optional[Set[str]] = None, hf_token: Optional[str] = None, allow_auto_repo: bool = True) -> dict:
    """Attempt to download missing checkpoint files from inferred HF repos.

    Tries a small set of candidate extensions for each expected checkpoint basename
    and returns the actual downloaded filenames in 'downloaded' and the attempted
    candidates that failed in 'failed'.
    """
    # If HF offline mode is set, do not attempt network downloads; return an informative error
    if os.environ.get('HF_HUB_OFFLINE') or os.environ.get('TRANSFORMERS_OFFLINE') or os.environ.get('HF_DATASETS_OFFLINE'):
        return {'downloaded': [], 'failed': list(expected or []), 'error': 'HF_HUB_OFFLINE set: refusing to perform network downloads in offline-only mode'}

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        return {'downloaded': [], 'failed': list(expected or []), 'error': 'huggingface_hub not available: ' + str(e)}

    pr = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    expected = expected or discover_expected_ckpts(pr)
    if not expected:
        return {'downloaded': [], 'failed': [], 'error': 'no expected ckpts found'}

    # source mapping
    src_map = discover_ckpt_sources(pr)
    ckpt_dir = os.path.join(model_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    token = hf_token or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')

    downloaded = []
    failed = []
    auth_required = False
    # preferred extension order
    candidates_ext = ('.safetensors', '.safetensor', '.pt', '.pth', '.ckpt')

    for stem in sorted(expected):
        # skip if any candidate already exists locally
        exists_locally = False
        for ext in candidates_ext:
            if os.path.exists(os.path.join(ckpt_dir, stem + ext)):
                exists_locally = True
                break
        if exists_locally:
            continue

        repo = src_map.get(stem) if allow_auto_repo else None
        if repo is None:
            repo = 'microsoft/TRELLIS-image-large'

        success = False
        for ext in candidates_ext:
            fn = stem + ext
            # Defensive sanity: repo should look like <owner>/<repo>, not 'ckpts/...'
            if repo.startswith('ckpts/') or repo.startswith('/ckpts'):
                # broken config: treat as missing mapping and use default repo
                repo = 'microsoft/TRELLIS-image-large'

            hf_path = os.path.join('ckpts', fn)
            try:
                cached = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
                import shutil
                dest = os.path.join(ckpt_dir, fn)
                shutil.copyfile(cached, dest)
                downloaded.append(fn)
                success = True
                break
            except Exception as e:
                msg = str(e).lower()
                if '401' in msg or 'unauthoriz' in msg or 'forbidden' in msg or 'token' in msg:
                    auth_required = True
                    failed.append(fn)
                    continue
                # fallback: try direct HTTP download from HF resolve URL (works for public files)
                hf_url = f'https://huggingface.co/{repo}/resolve/main/ckpts/{fn}'
                try:
                    import requests
                    with requests.get(hf_url, stream=True, timeout=30) as r:
                        if r.status_code == 200:
                            dest = os.path.join(ckpt_dir, fn)
                            with open(dest, 'wb') as fh:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        fh.write(chunk)
                            downloaded.append(fn)
                            success = True
                            break
                        else:
                            failed.append(fn)
                except Exception:
                    failed.append(fn)
        if not success:
            # no candidate succeeded for this stem
            pass

    result = {'downloaded': downloaded, 'failed': failed, 'auth_required': False}
    if auth_required and not token:
        result['auth_required'] = True
    return result

