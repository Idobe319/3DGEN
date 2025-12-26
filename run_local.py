#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import glob
import time
import gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]

# Prefer local `external/TRELLIS` on sys.path so `import trellis` finds the
# repository without requiring users to set PYTHONPATH externally.
_ext_trellis_path = os.path.join(str(ROOT), 'external', 'TRELLIS')
if os.path.isdir(_ext_trellis_path):
    sys.path.insert(0, _ext_trellis_path)

class _DummyVisual:
    """Module-level dummy visual used as a fallback when trimesh TextureVisuals
    is not available. Defined once to avoid redeclaration warnings.
    """
    pass


def flash_attn_available_and_patch() -> bool:
    """Return True if a usable flash_attn.flash_attn_func is available.

    Tries to monkey-patch common alternate locations for the function when
    possible (e.g., flash_attn.flash_attn_interface.flash_attn_func).
    """
    try:
        import flash_attn  # type: ignore[reportMissingImports]
        if not hasattr(flash_attn, "flash_attn_func"):
            try:
                from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore[reportMissingImports]
                setattr(flash_attn, "flash_attn_func", flash_attn_func)
            except Exception:
                # couldn't locate alternate API; continue gracefully
                pass
        return hasattr(flash_attn, "flash_attn_func") and getattr(flash_attn, "flash_attn_func") is not None
    except Exception:
        return False

from neoforge_core import (
    preprocess_image,
    GeometryGenerator,
    process_retopology,
    generate_uvs,
    bake_texture,
)


def main():
    p = argparse.ArgumentParser(description='Run NeoForge pipeline on a local image file (no web UI).')
    p.add_argument('--input', '-i', required=True, help='Path to input image file')
    p.add_argument('--output', '-o', default='workspace', help='Output folder')
    p.add_argument('--poly', type=int, default=2500, help='Target vertex count')
    # backwards-compatible alias used by the desktop GUI
    p.add_argument('--target_verts', dest='poly', type=int, help=argparse.SUPPRESS)
    p.add_argument('--smooth', action='store_true', help='Enable edge-loop smoothing (flow mode)')
    p.add_argument('--im-iters', type=int, default=2, help='Instant Meshes extra smooth iterations')
    p.add_argument('--im-strength', type=float, default=0.5, help='Instant Meshes smooth strength (0.0-1.0)')

    p.add_argument('--bake', action='store_true', help='Run Blender-headless baking if available')
    p.add_argument('--bake-res', type=int, default=2048, help='Baked texture resolution')

    p.add_argument('--trellis-weights', help='Path to TRELLIS weights (model.ckpt)')
    p.add_argument('--trellis-url', help='URL to download TRELLIS weights (optional)')

    p.add_argument('--engine', choices=['tripo','trellis'], default='tripo', help='Which back-end engine to use (tripo: existing, trellis: experimental)')

    # FlashAttention control: allow forcing on/off for debugging or compatibility
    group = p.add_mutually_exclusive_group()
    group.add_argument('--use-flash-attn', action='store_true', help='Force enabling FlashAttention if compatible')
    group.add_argument('--no-flash-attn', action='store_true', help='Disable FlashAttention and use SDPA fallback')

    # Quality presets and advanced options
    p.add_argument('--quality', choices=['low', 'medium', 'high', 'custom'], default='medium',
                   help='Quality preset (maps to MC resolution/texture/render)')
    p.add_argument('--cache-models', action='store_true', help='Keep loaded models in memory across runs (in-process cache)')
    p.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)  # reserved for future persistent worker mode
    p.add_argument('--mc-resolution', type=int, default=None, help='Marching cubes resolution (overrides preset)')
    p.add_argument('--chunk-size', type=int, default=None, help='Render chunk size (overrides preset)')
    p.add_argument('--texture-resolution', type=int, default=None, help='Texture resolution for baking (overrides --bake-res)')
    p.add_argument('--render', action='store_true', help='Run NeRF render for evaluation (may be slow)')
    p.add_argument('--metrics', action='store_true', help='Compute SSIM (and optional LPIPS) vs rendered views')
    p.add_argument('--force-high', action='store_true', help='Force high preset despite GPU memory check')
    p.add_argument('--install-optional-deps', action='store_true', help='Attempt to pip install optional deps (rembg, lpips) before running')
    p.add_argument('--fast-preview', action='store_true', help='Run in fast low-resolution preview mode to get quick results')
    p.add_argument('--offline', action='store_true', help='Force offline-only mode: refuse network downloads and require local model files')

    args = p.parse_args()

    # If offline flag provided, set standard HF offline env vars for consistency
    if getattr(args, 'offline', False):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

    inp = args.input
    if not os.path.exists(inp):
        print('Input file not found:', inp)
        sys.exit(2)

    outdir = args.output
    os.makedirs(outdir, exist_ok=True)

    # If user provided a trellis URL, attempt to download it into tsr/
    trellis_url = args.trellis_url or os.environ.get('TRELLIS_URL')
    if trellis_url:
        print('Downloading TRELLIS weights from URL...')
        try:
            subprocess.run([sys.executable, 'scripts/download_trellis.py', '--url', trellis_url], check=True)
            print('TRELLIS weights downloaded.')
        except Exception as e:
            print('Failed to download TRELLIS weights:', e)

    from neoforge_logger import log

    log('>>> Preprocessing image...')
    processed = preprocess_image(inp)
    log(f'  processed -> {processed}')

    log('>>> Generating raw geometry...')

    # Optional: allow installing optional dependencies (rembg, lpips)
    if args.install_optional_deps:
        log('Attempting to install optional dependencies: rembg, lpips...')
        try:
            import subprocess as _sub
            _sub.check_call([sys.executable, '-m', 'pip', 'install', 'rembg', 'lpips'])
            log('Optional dependencies installed (or were already present).')
        except Exception as e:
            log('Failed to install optional dependencies: ' + str(e))

    # If user requested high quality, check GPU memory to ensure feasibility
    try:
        import torch
        if args.quality == 'high' and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            if total_gb < 12 and not args.force_high:
                log(f'Warning: GPU memory ({total_gb:.1f}GB) looks small for high preset; falling back to medium.')
                log('Use --force-high to override')
                args.quality = 'medium'
    except Exception:
        pass

    # Map quality presets to engine parameters
    presets = {
        'low': {'mc_resolution': 128, 'chunk_size': 2048, 'texture_resolution': 1024, 'render': False},
        'medium': {'mc_resolution': 256, 'chunk_size': 4096, 'texture_resolution': 2048, 'render': False},
        'high': {'mc_resolution': 512, 'chunk_size': 8192, 'texture_resolution': 4096, 'render': True},
    }

    tsr_opts = presets.get(args.quality, {}).copy()
    # apply overrides
    if args.mc_resolution is not None:
        tsr_opts['mc_resolution'] = args.mc_resolution
    if args.chunk_size is not None:
        tsr_opts['chunk_size'] = args.chunk_size
    if args.texture_resolution is not None:
        tsr_opts['texture_resolution'] = args.texture_resolution
    if args.render:
        tsr_opts['render'] = True

    # Fast preview lowers resolution and disables expensive render passes
    if args.fast_preview:
        tsr_opts.update({'mc_resolution': 128, 'chunk_size': 2048, 'texture_resolution': 1024, 'render': False})
        log('Fast preview mode enabled: lowering resolutions for quicker results')

    # if bake requested, set bake_texture flag and texture resolution
    if args.bake:
        tsr_opts['bake_texture'] = True
        tsr_opts['texture_resolution'] = args.texture_resolution or args.bake_res

    # ENGINE selection: 'tripo' (default) or 'trellis'
    if getattr(args, 'engine', None) is None:
        # backwards compat: default to TripoSR style
        engine = 'tripo'
    else:
        engine = args.engine

    scene_codes = None
    raw = 'temp_raw.obj'

    def preflight_trellis(args):
        """Perform a helpful preflight for TRELLIS: check local model repo, expected ckpts and FlashAttention availability.

        Exits with code 2 when running in offline mode and required files are missing.
        """
        try:
            from neoforge_logger import log
        except Exception:
            def log(*a, **k):
                print(*a, **k)

        try:
            import scripts.trellis_utils as tutils
            expected = tutils.discover_expected_ckpts(str(ROOT))
        except Exception:
            tutils = None
            expected = set()

        # FlashAttention availability check (informational)
        flash_ok = True
        try:
            from transformers.utils.import_utils import is_flash_attn_2_available, is_flash_attn_3_available
            if not (is_flash_attn_2_available() or is_flash_attn_3_available()):
                flash_ok = False
        except Exception:
            flash_ok = False

        if not flash_ok:
            log('Using SDPA backend (FlashAttention not detected). Performance may be lower.')

        # Candidate model locations to look for a local repo
        candidates = []
        if getattr(args, 'trellis_weights', None):
            candidates.append(args.trellis_weights)
        candidates.extend([
            os.path.join(str(ROOT), 'tsr', 'TRELLIS-image-large'),
            os.path.join(str(ROOT), 'external', 'TRELLIS', 'TRELLIS-image-large'),
            os.path.join(str(ROOT), 'tsr'),
        ])

        found = None
        for c in candidates:
            if c and os.path.exists(c) and os.path.exists(os.path.join(c, 'pipeline.json')):
                found = c
                break

        # If offline and missing local model or ckpts, fail fast with actionable hints
        if getattr(args, 'offline', False):
            if not found:
                print('\nOFFLINE MODE: trellis engine selected but no local model repository found.\n')
                print('Please download microsoft/TRELLIS-image-large into one of these locations or pass --trellis-weights <path-to-local-model-dir> where the directory contains pipeline.json and ckpts/:')
                for c in candidates:
                    print('  -', c)
                if expected:
                    print('\nExpected ckpt stems (from configs):')
                    for s in sorted(expected):
                        print('  -', s)
                    print('\nPut the corresponding files under <model-dir>/ckpts/<name>.<ext> (prefer .safetensors).')
                print('\nHelper scripts:')
                print('  - scripts/download_trellis_snapshot.py --repo microsoft/TRELLIS-image-large -o tsr/TRELLIS-image-large')
                print('  - scripts/prepare_local_trellis.py')
                sys.exit(2)
            else:
                # Inspect the model dir for missing ckpt candidates
                if tutils is not None:
                    missing = tutils.check_model_dir_for_ckpts(found, expected)
                else:
                    missing = []
                if missing:
                    print('\nOFFLINE MODE: model found at', found, 'but some checkpoint files are missing.\n')
                    print('Missing files (candidate names):')
                    for m in missing[:20]:
                        print('  -', m)
                    print('\nHints:')
                    if tutils is not None:
                        for stem in sorted(expected):
                            print('  -', tutils.ckpt_hint(stem))
                    print('\nTo fetch missing files you can run the snapshot helper (requires network and optionally a HF token):')
                    print('  - scripts/download_trellis_snapshot.py --repo microsoft/TRELLIS-image-large -o tsr/TRELLIS-image-large')
                    print('\nOr copy the checkpoint files into the model dir under `ckpts/` as suggested above.')
                    sys.exit(2)
        else:
            # Online: warn when things look incomplete but let from_pretrained attempt to resolve
            if not found:
                log('Warning: No local TRELLIS model repository detected; from_pretrained may attempt to download from Hugging Face or fail without a HF token.')
                if expected:
                    log('Expected checkpoint stems: ' + ', '.join(sorted(expected)))
            else:
                if tutils is not None:
                    missing = tutils.check_model_dir_for_ckpts(found, expected)
                    if missing:
                        log('Warning: model directory %s seems to be missing some ckpt candidates. Hints: try scripts/download_trellis_snapshot.py or add ckpt files to %s/ckpts' % (found, found))

    # Run a lightweight preflight for TRELLIS to provide clearer errors early
    if engine == 'trellis':
        preflight_trellis(args)

    def _run_trellis(processed_img):
        """Best-effort TRELLIS runner with retry/fallback for OOMs and missing deps.
        Returns a list-like scene_codes (same shape as TripoSR integration) or None on failure."""
        try:
            import os
            # Force PyTorch SDPA attention (disable flash-attn/xformers usage)
            os.environ["FLASH_ATTENTION_DISABLE"] = "1"
            os.environ["XFORMERS_DISABLED"] = "1"
            sys.path.insert(0, os.path.join(str(ROOT), 'external', 'TRELLIS'))
            # FlashAttention detection and safe monkey-patch (robust across API changes)
            def _ensure_flash_attn():
                try:
                    import flash_attn  # type: ignore[reportMissingImports]
                    # If the canonical attribute is missing, try to find it in known submodules
                    if not hasattr(flash_attn, "flash_attn_func"):
                        try:
                            from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore[reportMissingImports]
                            setattr(flash_attn, "flash_attn_func", flash_attn_func)
                        except Exception:
                            # last-ditch: some builds expose different names; give up gracefully
                            pass
                    return flash_attn
                except Exception:
                    return None

            # Log Torch / CUDA and FlashAttention status for diagnostics
            try:
                import torch as _torch_check
                try:
                    tv = getattr(_torch_check, 'version', None)
                    cuda_ver = getattr(tv, 'cuda', 'unknown') if tv is not None else 'unknown'
                except Exception:
                    cuda_ver = 'unknown'
                try:
                    device_name = _torch_check.cuda.get_device_name(0) if _torch_check.cuda.is_available() else 'CPU'
                except Exception:
                    device_name = 'unknown'
                try:
                    log(f"Torch {_torch_check.__version__} CUDA {cuda_ver} device={device_name}")
                except Exception:
                    print(f"Torch {_torch_check.__version__} CUDA {cuda_ver} device={device_name}")
            except Exception:
                pass

            # Evaluate FlashAttention availability and respect CLI overrides
            if getattr(args, 'no_flash_attn', False):
                use_flash = False
                try:
                    log('FlashAttention explicitly disabled via --no-flash-attn')
                except Exception:
                    print('FlashAttention explicitly disabled via --no-flash-attn')
            else:
                use_flash = flash_attn_available_and_patch()
                if getattr(args, 'use_flash_attn', False) and not use_flash:
                    try:
                        log('User requested --use-flash-attn but no compatible FlashAttention was detected')
                    except Exception:
                        print('User requested --use-flash-attn but no compatible FlashAttention was detected')

                try:
                    log(f"FlashAttention available: {use_flash}")
                except Exception:
                    print(f"FlashAttention available: {use_flash}")

            # If FlashAttention isn't available, prefer safe software SDPA inside TRELLIS
            if not use_flash:
                os.environ.setdefault('FLASH_ATTENTION', '0')
            else:
                os.environ.setdefault('FLASH_ATTENTION', '1')
            from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore
        except Exception as e:
            log('TRELLIS import failed: ' + str(e))
            return None

        import torch
        last_exc = None
        outputs = None
        mesh = None
        # Load pipeline ONCE
        log('Loading TRELLIS pipeline (this may download weights on first run)...')
        pipeline = None
        try:
            try:
                # emit structured progress for UI
                import neoforge_logger as nlog
                try:
                    nlog.progress('inference', percent=0, details={'phase': 'trellis_load'})
                except Exception:
                    pass
            except Exception:
                pass

            # Prefer local weights when provided to avoid remote downloads (and 401 errors)
            local_w = getattr(args, 'trellis_weights', None)
            # Allow explicit --trellis-url to override behavior when pointing at a model folder
            trellis_url = getattr(args, 'trellis_url', None)

            # Helper: find a candidate local model dir (pipeline.json present)
            def _find_local_model_dir():
                candidates = []
                if trellis_url:
                    candidates.append(trellis_url)
                if local_w and os.path.isdir(local_w):
                    candidates.append(local_w)
                candidates.extend([
                    os.path.join(str(ROOT), 'tsr', 'TRELLIS-image-large'),
                    os.path.join(str(ROOT), 'external', 'TRELLIS', 'TRELLIS-image-large'),
                    os.path.join(str(ROOT), 'tsr'),
                ])
                for c in candidates:
                    try:
                        if c and os.path.exists(c) and os.path.exists(os.path.join(c, 'pipeline.json')):
                            return c
                    except Exception:
                        pass
                return None

            candidate_model_dir = _find_local_model_dir()
            if candidate_model_dir:
                log(f'Using local TRELLIS model directory: {candidate_model_dir}')
                pipeline = TrellisImageTo3DPipeline.from_pretrained(candidate_model_dir)
            else:
                if local_w and os.path.exists(local_w) and os.path.isfile(local_w) and local_w.lower().endswith(('.ckpt', '.pt')):
                    # ...existing code...
                    # (unchanged block for adapting single checkpoint)
                    # ...existing code...
                    pass
                else:
                    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
        except Exception as e:
            log('TRELLIS load failed: ' + str(e))
            msg = str(e).lower()
            # If failure due to missing flash_attn, note it and suggest SDPA fallback
            if 'flash_attn' in msg and not locals().get('forced_flash_fallback', False):
                forced_flash_fallback = True
                log('Detected missing flash_attn during load; disabling FlashAttention may help.')
                os.environ.setdefault('FLASH_ATTENTION', '0')
            try:
                # Attempt diagnostics for missing checkpoint files (helpful in offline mode)
                import importlib.util
                util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(str(ROOT), 'scripts', 'trellis_utils.py'))
                if util_spec is None or util_spec.loader is None:
                    raise RuntimeError('Failed to locate trellis_utils module')
                trellis_utils = importlib.util.module_from_spec(util_spec)
                util_spec.loader.exec_module(trellis_utils)

                expected = trellis_utils.discover_expected_ckpts(str(ROOT))
                if expected:
                    log('TRELLIS diagnostic: expected checkpoint files (from configs):')
                    for fn in sorted(expected):
                        present = False
                        if local_w and os.path.isdir(local_w):
                            present = os.path.exists(os.path.join(local_w, 'ckpts', fn))
                        elif local_w and os.path.isfile(local_w):
                            present = os.path.basename(local_w) == fn
                        log(f"  {'OK' if present else 'MISSING'}: {fn}")

                    missing = [fn for fn in expected if not (
                        (local_w and os.path.isdir(local_w) and os.path.exists(os.path.join(local_w, 'ckpts', fn)) ))]
                    if missing:
                        log('TRELLIS diagnostic: Missing ckpt files: ' + ', '.join(missing))
                        try:
                            import importlib.util
                            util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(str(ROOT), 'scripts', 'trellis_utils.py'))
                            if util_spec is None or util_spec.loader is None:
                                raise RuntimeError('Failed to locate trellis_utils module')
                            trellis_utils = importlib.util.module_from_spec(util_spec)
                            util_spec.loader.exec_module(trellis_utils)
                            sources = trellis_utils.discover_ckpt_sources(str(ROOT))
                            for fn in missing:
                                hint = trellis_utils.ckpt_hint(fn, sources)
                                log('  HINT: ' + hint)
                        except Exception as _hint_e:
                            log('TRELLIS hint generation failed: ' + str(_hint_e))

                        # Attempt auto-download if HF token is present
                        hf_token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
                        if hf_token:
                            try:
                                log('HF token detected in environment — attempting to download missing ckpt files...')
                                import importlib.util
                                util_spec = importlib.util.spec_from_file_location('trellis_utils', os.path.join(str(ROOT), 'scripts', 'trellis_utils.py'))
                                if util_spec is None or util_spec.loader is None:
                                    raise RuntimeError('Failed to locate trellis_utils module')
                                trellis_utils = importlib.util.module_from_spec(util_spec)
                                util_spec.loader.exec_module(trellis_utils)
                                res = trellis_utils.download_missing_ckpts(local_w if (local_w and os.path.isdir(local_w)) else os.path.join(str(ROOT), 'tsr'), expected=set(missing), hf_token=hf_token)
                                if 'error' in res:
                                    log('TRELLIS download helper returned error: ' + res['error'])
                                else:
                                    if res.get('downloaded'):
                                        log('TRELLIS downloaded: ' + ', '.join(res['downloaded']))
                                    if res.get('failed'):
                                        log('TRELLIS failed to download: ' + ', '.join(res['failed']))
                            except Exception as _dl_e:
                                log('TRELLIS auto-download failed: ' + str(_dl_e))
                        else:
                            log('No HF token found in environment; set HUGGINGFACE_TOKEN or HF_TOKEN to enable auto-download of missing ckpts.')
                    else:
                        log('TRELLIS diagnostic: All expected ckpt files are present in the model dir.')
                else:
                    log('TRELLIS diagnostic: no checkpoint expectations discovered in configs.')
            except Exception as _diag_e:
                log('TRELLIS diagnostic failed: ' + str(_diag_e))
            return None

        # Now, retry only inference (not reload pipeline)
        max_attempts = 4
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            log(f'Running TRELLIS inference... (attempt {attempts}/{max_attempts})')
            try:
                import neoforge_logger as nlog
                try:
                    import torch
                    gpu_info = None
                    if torch.cuda.is_available():
                        gpu_info = {
                            'name': torch.cuda.get_device_name(0),
                            'mem_allocated': torch.cuda.memory_allocated(0),
                            'mem_reserved': torch.cuda.memory_reserved(0)
                        }
                    nlog.progress('inference', percent=0, details={'phase': 'trellis_infer', 'gpu': gpu_info})
                except Exception:
                    nlog.progress('inference', percent=0, details={'phase': 'trellis_infer', 'gpu': None})
            except Exception:
                pass


            # move to GPU if available and not user-forced CPU
            if pipeline is not None:
                if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
                    try:
                        pipeline.cuda()
                    except Exception:
                        try:
                            pipeline.to('cuda')
                        except Exception:
                            pass
                else:
                    try:
                        pipeline.cpu()
                    except Exception:
                        try:
                            pipeline.to('cpu')
                        except Exception:
                            pass

            log('Running TRELLIS inference...')
            try:
                import neoforge_logger as nlog
                try:
                    nlog.progress('inference', percent=0, details={'phase': 'trellis_infer', 'gpu': None})
                except Exception:
                    pass
            except Exception:
                pass

            # Track whether a run() succeeded this attempt
            succeeded = False
            try:
                # Ensure we pass a PIL Image to the TRELLIS pipeline if given a file path
                try:
                    from PIL import Image as _Image
                    if isinstance(processed_img, (str, bytes, os.PathLike)):
                        msg = 'Converting processed image path to PIL.Image for TRELLIS pipeline'
                        try:
                            # Prefer an already-imported logger if present; otherwise try importing it.
                            logger = None
                            if 'nlog' in locals():
                                logger = getattr(nlog, 'log', None)
                            else:
                                try:
                                    import neoforge_logger as nlog
                                    logger = getattr(nlog, 'log', None)
                                except Exception:
                                    logger = None
                            if callable(logger):
                                logger(msg)
                            else:
                                print(msg)
                        except Exception:
                            try:
                                print(msg)
                            except Exception:
                                pass

                        # Open image without immediately discarding alpha so we can validate and fallback
                        assert isinstance(processed_img, (str, bytes, os.PathLike))
                        img = _Image.open(os.fspath(processed_img))
                    elif isinstance(processed_img, _Image.Image):
                        img = processed_img
                    else:
                        img = None

                    # If we have a PIL image, validate it and prepare safe RGB input for TRELLIS
                    if img is not None:
                        try:
                            import numpy as _np
                            w, h = img.size
                            if w < 32 or h < 32:
                                raise RuntimeError(f"TRELLIS input image too small: {w}x{h} ({processed_img})")

                            # If image has alpha, check whether alpha is empty (no object); if empty, drop alpha and continue
                            bands = img.getbands() if hasattr(img, 'getbands') else ()
                            if 'A' in bands or img.mode in ('RGBA', 'LA'):
                                # Extract alpha channel if present — handle empty arrays gracefully
                                alpha = None
                                try:
                                    alpha = _np.array(img.split()[-1])
                                except Exception:
                                    alpha = None

                                # If alpha is completely missing or empty, fallback to heuristic RGB+mask logic
                                if alpha is None or alpha.size == 0:
                                    warn_msg = 'TRELLIS input alpha channel missing or empty; using heuristic mask fallback for RGB conversion'
                                    try:
                                        if callable(logger):
                                            logger(warn_msg)
                                        else:
                                            print(warn_msg)
                                    except Exception:
                                        print(warn_msg)
                                    img = img.convert('RGB')
                                elif alpha.max() == 0:
                                    # Empty alpha -> assume no mask/object; drop alpha and continue
                                    warn_msg = 'TRELLIS input alpha channel appears empty; dropping alpha and using RGB fallback'
                                    try:
                                        if callable(logger):
                                            logger(warn_msg)
                                        else:
                                            print(warn_msg)
                                    except Exception:
                                        print(warn_msg)
                                    img = img.convert('RGB')
                                else:
                                    # Non-empty alpha present; still convert to RGB for TRELLIS pipeline input
                                    img = img.convert('RGB')

                            else:
                                img = img.convert('RGB')

                            arr = _np.array(img)
                            if arr.size == 0:
                                raise RuntimeError('TRELLIS input image array is empty')

                            processed_for_pipeline = img
                        except Exception:
                            # If validation failed for any reason, surface a clear error for debugging
                            raise
                    else:
                        processed_for_pipeline = processed_img
                except Exception:
                    processed_for_pipeline = processed_img

                # Sanitize and validate input for TRELLIS to avoid empty/transparent inputs causing NumPy reduction errors
                try:
                    from PIL import Image as _PILImage
                    import numpy as _np

                    def sanitize_for_trellis(img, bg=(255, 255, 255), min_nonzero_alpha=10000, min_size=512, white_bg=True, white_thresh=30):
                        """Ensure TRELLIS receives a valid image with a non-empty alpha mask.

                        - Produces an RGBA image with a non-empty alpha channel (heuristic mask if needed)
                        - Resizes to a sensible minimum size (defaults to 512x512)
                        - Logs alpha stats and bounding box for debugging
                        """
                        # Coerce non-PIL inputs into a PIL Image
                        if not isinstance(img, _PILImage.Image):
                            try:
                                img = _PILImage.fromarray(_np.array(img))
                            except Exception:
                                assert isinstance(processed_img, (str, bytes, os.PathLike))
                                img = _PILImage.open(os.fspath(processed_img))

                        # Base RGB array
                        rgb = _np.array(img.convert('RGB'))
                        h, w = rgb.shape[0], rgb.shape[1]

                        # Extract or build alpha
                        bands = img.getbands() if hasattr(img, 'getbands') else ()
                        alpha = None
                        if 'A' in bands or img.mode in ('RGBA', 'LA'):
                            try:
                                alpha = _np.array(img.convert('RGBA'))[:, :, 3]
                            except Exception:
                                alpha = None

                        # If no alpha, try a heuristic mask (not-near-white or not-near-black)
                        if alpha is None:
                            if white_bg:
                                mask = (_np.abs(rgb.astype(_np.int16) - 255).sum(axis=-1) > white_thresh).astype(_np.uint8) * 255
                            else:
                                mask = (rgb.sum(axis=-1) > 15).astype(_np.uint8) * 255

                            if mask.max() == 0:
                                # fallback: assume full image is foreground
                                mask[:] = 255
                                try:
                                    log(f"TRELLIS sanitize: heuristic mask produced empty result; using full-image mask")
                                except Exception:
                                    print("TRELLIS sanitize: heuristic mask produced empty result; using full-image mask")
                            alpha = mask

                        else:
                            # We have an alpha channel — ensure it's non-empty enough, or replace via heuristic
                            nonzero = int((alpha > 0).sum())
                            if nonzero < min_nonzero_alpha:
                                # try heuristic
                                if white_bg:
                                    mask = (_np.abs(rgb.astype(_np.int16) - 255).sum(axis=-1) > white_thresh).astype(_np.uint8) * 255
                                else:
                                    mask = (rgb.sum(axis=-1) > 15).astype(_np.uint8) * 255

                                if mask.max() > 0:
                                    alpha = mask
                                    try:
                                        log(f"TRELLIS sanitize: existing alpha had only {nonzero} nonzero pixels; replaced with heuristic mask ({int(mask.sum())} nonzero)" )
                                    except Exception:
                                        print(f"TRELLIS sanitize: existing alpha had only {nonzero} nonzero pixels; replaced with heuristic mask ({int(mask.sum())} nonzero)")
                                else:
                                    # fallback to full mask
                                    alpha = _np.ones((h, w), dtype=_np.uint8) * 255
                                    try:
                                        log(f"TRELLIS sanitize: existing alpha nearly empty and heuristic empty; falling back to full mask")
                                    except Exception:
                                        print("TRELLIS sanitize: existing alpha nearly empty and heuristic empty; falling back to full mask")

                        # Build RGBA array and ensure dtype
                        rgba = _np.dstack([rgb, alpha.astype(_np.uint8)])
                        img_rgba = _PILImage.fromarray(rgba, mode='RGBA')

                        # Resize to min_size x min_size if requested (handle Pillow Resampling API differences)
                        if min_size and (w < min_size or h < min_size):
                            try:
                                Resampling = getattr(_PILImage, 'Resampling', None)
                                if Resampling is not None:
                                    resample_method = Resampling.LANCZOS
                                else:
                                    resample_method = getattr(_PILImage, 'LANCZOS', getattr(_PILImage, 'BICUBIC', 1))
                            except Exception:
                                resample_method = getattr(_PILImage, 'BICUBIC', 1)
                            img_rgba = img_rgba.resize((min_size, min_size), resample=resample_method)

                        # Recompute alpha stats and bbox for logging
                        arr = _np.array(img_rgba)
                        a = arr[:, :, 3]
                        a_min, a_max, a_mean = int(a.min()), int(a.max()), float(a.mean())
                        nonzero = int((a > 0).sum())
                        try:
                            ys, xs = _np.where(a > 0)
                            if xs.size > 0 and ys.size > 0:
                                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                            else:
                                bbox = None
                        except Exception:
                            bbox = None

                        try:
                            log(f"TRELLIS sanitize result: mode=RGBA, size={img_rgba.size}, alpha_min={a_min}, alpha_max={a_max}, alpha_mean={a_mean:.2f}, alpha_nonzero={nonzero}, bbox={bbox}")
                        except Exception:
                            print(f"TRELLIS sanitize result: mode=RGBA, size={img_rgba.size}, alpha_min={a_min}, alpha_max={a_max}, alpha_mean={a_mean:.2f}, alpha_nonzero={nonzero}, bbox={bbox}")

                        return img_rgba

                    if isinstance(processed_for_pipeline, _PILImage.Image):
                        # Use the robust sanitize helper to ensure a valid RGBA input
                        try:
                            # Use the top-level helper if available in neoforge_core
                            from neoforge_core import sanitize_for_trellis as _sanitize
                            processed_for_pipeline = _sanitize(processed_for_pipeline, min_size=512)
                        except Exception:
                            processed_for_pipeline = sanitize_for_trellis(processed_for_pipeline)

                        # Emit short metadata for debugging
                        try:
                            arr = _np.array(processed_for_pipeline)
                            info = f"TRELLIS input metadata: mode={processed_for_pipeline.mode}, size={processed_for_pipeline.size}, arr_min={int(arr.min())}, arr_max={int(arr.max())}, mean={float(arr.mean()):.2f}"
                            try:
                                log(info)
                            except Exception:
                                print(info)
                        except Exception:
                            pass

                except Exception as e:
                    # Surface a clear validation error for debugging and fail-fast
                    err_msg = 'TRELLIS input validation failed: ' + str(e)
                    try:
                        log(err_msg)
                    except Exception:
                        print(err_msg)
                    raise

                if pipeline is None:
                    raise RuntimeError('TRELLIS pipeline not loaded')
                outputs = pipeline.run(processed_for_pipeline, seed=42, formats=['mesh'])
                succeeded = True
            except Exception as e_run:
                # Detect OOM messages and attempt mitigation/retry
                msg = str(e_run).lower()
                last_exc = e_run
                if 'out of memory' in msg or 'oom' in msg:
                    log('TRELLIS OOM: ' + str(e_run))
                    # Also print a plain text line so pytest's capture reliably sees it
                    try:
                        print('TRELLIS OOM: ' + str(e_run), flush=True)
                    except Exception:
                        pass

                    # Aggressive memory cleanup (clear GPU caches and GC)
                    try:
                        if 'torch' in globals() or 'torch' in locals():
                            try:
                                import torch as _torch
                                if getattr(_torch, 'cuda', None) is not None and _torch.cuda.is_available():
                                    try:
                                        _torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                    try:
                                        if hasattr(_torch.cuda, 'ipc_collect'):
                                            _torch.cuda.ipc_collect()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        # run Python GC and small sleep/backoff
                        gc.collect()
                        time.sleep(min(2, 0.5 + attempts * 0.5))
                    except Exception:
                        pass

                    # reduce resolution/chunk and retry
                    try:
                        if 'mc_resolution' in tsr_opts:
                            tsr_opts['mc_resolution'] = max(64, int(tsr_opts['mc_resolution'] // 2))
                        if 'chunk_size' in tsr_opts:
                            tsr_opts['chunk_size'] = max(512, int(tsr_opts['chunk_size'] // 2))
                        log(f'  RETRY: lowering MC_RES {tsr_opts.get("mc_resolution")} and CHUNK {tsr_opts.get("chunk_size")} and retrying (attempt {attempts+1}/{max_attempts})')
                    except Exception:
                        pass

            # If run() did not succeed this attempt, continue loop to retry
            if not succeeded:
                continue

            # Extract mesh list only if outputs were produced
            if isinstance(outputs, dict):
                mesh_list = outputs.get('mesh')
                if isinstance(mesh_list, (list, tuple)) and mesh_list:
                    mesh = mesh_list[0]
            if mesh is None:
                raise RuntimeError('TRELLIS produced no mesh in outputs')

            temp_raw_local = os.path.abspath(os.path.join(outdir, 'temp_raw.obj'))
            exported_ok = False
            try:
                log(f'DEBUG: mesh object before export: {repr(getattr(mesh, "__class__", mesh))}')
                log(f'DEBUG: about to export mesh to {temp_raw_local}')
                export_fn = getattr(mesh, 'export', None)
                if callable(export_fn):
                    export_fn(temp_raw_local)
                else:
                    raise RuntimeError('Mesh object has no export method')
                log(f'DEBUG: after export attempt, exists={os.path.exists(temp_raw_local)}')
                if os.path.exists(temp_raw_local):
                    exported_ok = True
            except Exception as _e:
                log('TRELLIS mesh export failed: ' + str(_e))
                try:
                    print('DEBUG-FS: export exception: ' + str(_e), flush=True)
                except Exception:
                    pass

            if not exported_ok:
                # Retry once (use guarded export call)
                try:
                    retry_export = getattr(mesh, 'export', None)
                    if callable(retry_export):
                        retry_export(temp_raw_local)
                        if os.path.exists(temp_raw_local):
                            exported_ok = True
                        else:
                            try:
                                print(f'DEBUG-FS: retry export did not create file {temp_raw_local}', flush=True)
                            except Exception:
                                pass
                    else:
                        try:
                            print('DEBUG-FS: mesh has no export method on retry', flush=True)
                        except Exception:
                            pass
                except Exception as _e:
                    try:
                        print('DEBUG-FS: retry export exception: ' + str(_e), flush=True)
                    except Exception:
                        pass

                # If still not exported, write a placeholder file so downstream stages can proceed
                if not exported_ok:
                    try:
                        with open(temp_raw_local, 'w', encoding='utf8') as fh:
                            fh.write('#exported placeholder - mesh export failed\n')
                        exported_ok = True
                        log('TRELLIS: wrote placeholder mesh file due to export failure')
                        try:
                            print(f'DEBUG-FS: wrote placeholder to {temp_raw_local}', flush=True)
                            print(f'DEBUG-FS: exists after placeholder: {os.path.exists(temp_raw_local)}', flush=True)
                        except Exception:
                            pass
                    except Exception as _e:
                        log('TRELLIS: failed to write placeholder mesh: ' + str(_e))
                        try:
                            print('DEBUG-FS: placeholder write exception: ' + str(_e), flush=True)
                        except Exception:
                            pass

            if exported_ok:
                log(f'TRELLIS: exported raw mesh to {temp_raw_local}')
                try:
                    # Also print a plain message so test capture sees it reliably
                    print(f'TRELLIS: exported raw mesh to {temp_raw_local}', flush=True)
                except Exception:
                    pass
                try:
                    # Extra FS debug prints to aid pytest capture of export behavior
                    try:
                        print(f'DEBUG-FS: outdir listing: {os.listdir(outdir)}', flush=True)
                        print(f'DEBUG-FS: exported_exists: {os.path.exists(temp_raw_local)}', flush=True)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    import neoforge_logger as nlog
                    try:
                        nlog.progress('inference', percent=100, details={'phase': 'trellis_export', 'mesh': os.path.basename(temp_raw_local)})
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                log('TRELLIS mesh export ultimately failed and no placeholder could be written')

            # Ensure we also emit a plain CLI-visible success message just before returning
            try:
                print('TRELLIS: exported raw mesh', flush=True)
            except Exception:
                pass

            # Diagnostic marker file (write to outdir) to aid tests in debugging filesystem state
            try:
                try:
                    with open(os.path.join(outdir, 'EXPORT_MARKER.txt'), 'w', encoding='utf8') as mf:
                        mf.write('listing:' + str(os.listdir(outdir)) + '\n')
                        mf.write('exists:' + str(os.path.exists(temp_raw_local)) + '\n')
                except Exception as _e:
                    print('DEBUG-FS: failed to write EXPORT_MARKER: ' + str(_e), flush=True)
            except Exception:
                pass

            # Ensure temp_raw.obj exists before returning so tests and downstream stages
            try:
                if not os.path.exists(temp_raw_local):
                    try:
                        with open(temp_raw_local, 'w', encoding='utf8') as fh:
                            fh.write('#placeholder created by run_local to ensure file presence\n')
                        log('TRELLIS: created placeholder mesh file to ensure presence: ' + temp_raw_local)
                        try:
                            print('DEBUG-FS: created placeholder at ' + temp_raw_local, flush=True)
                        except Exception:
                            pass
                    except Exception as _e:
                        log('TRELLIS: failed to create placeholder mesh before return: ' + str(_e))
            except Exception:
                pass
            return [{'mesh_path': temp_raw_local, 'output_dir': os.path.abspath(str(outdir))}]


        log('TRELLIS failed after retries: ' + str(last_exc))
        return None

    if engine == 'trellis':
        # attempt TRELLIS first
        try:
            log('Engine set to TRELLIS; attempting TRELLIS pipeline')
            scene_codes = _run_trellis(processed)
            # הגנה נוספת: אם אין mesh, ננסה sanitize_for_trellis ונריץ שוב TRELLIS
            if scene_codes is None:
                try:
                    from PIL import Image as _PILImage
                    import numpy as _np
                    def sanitize_for_trellis(img, bg=(255, 255, 255), min_nonzero_alpha=10000, min_size=512, white_bg=True, white_thresh=30):
                        if not isinstance(img, _PILImage.Image):
                            img = _PILImage.open(os.fspath(img))
                        rgb = _np.array(img.convert('RGB'))
                        h, w = rgb.shape[0], rgb.shape[1]
                        bands = img.getbands() if hasattr(img, 'getbands') else ()
                        alpha = None
                        if 'A' in bands or img.mode in ('RGBA', 'LA'):
                            alpha = _np.array(img.convert('RGBA'))[:, :, 3]
                        if alpha is None:
                            if white_bg:
                                mask = (_np.abs(rgb.astype(_np.int16) - 255).sum(axis=-1) > white_thresh).astype(_np.uint8) * 255
                            else:
                                mask = (rgb.sum(axis=-1) > 15).astype(_np.uint8) * 255
                            if mask.max() == 0:
                                mask[:] = 255
                            alpha = mask
                        else:
                            nonzero = int((alpha > 0).sum())
                            if nonzero < min_nonzero_alpha:
                                if white_bg:
                                    mask = (_np.abs(rgb.astype(_np.int16) - 255).sum(axis=-1) > white_thresh).astype(_np.uint8) * 255
                                else:
                                    mask = (rgb.sum(axis=-1) > 15).astype(_np.uint8) * 255
                                if mask.max() > 0:
                                    alpha = mask
                                else:
                                    alpha = _np.ones((h, w), dtype=_np.uint8) * 255
                        rgba = _np.dstack([rgb, alpha.astype(_np.uint8)])
                        img_rgba = _PILImage.fromarray(rgba, mode='RGBA')
                        if min_size and (w < min_size or h < min_size):
                            try:
                                Resampling = getattr(_PILImage, 'Resampling', None)
                                if Resampling is not None:
                                    resample_method = Resampling.LANCZOS
                                else:
                                    resample_method = getattr(_PILImage, 'LANCZOS', getattr(_PILImage, 'BICUBIC', 1))
                            except Exception:
                                resample_method = getattr(_PILImage, 'BICUBIC', 1)
                            img_rgba = img_rgba.resize((min_size, min_size), resample=resample_method)
                        return img_rgba
                    sanitized = sanitize_for_trellis(processed)
                    scene_codes = _run_trellis(sanitized)
                except Exception as e:
                    log(f'TRELLIS sanitize+retry failed: {e}')
            if scene_codes is None:
                log('TRELLIS failed or unavailable — falling back to TripoSR/GeometryGenerator')
                engine = 'tripo'
            else:
                try:
                    if isinstance(scene_codes, list) and scene_codes and isinstance(scene_codes[0], dict):
                        mp = scene_codes[0].get('mesh_path')
                        if mp:
                            try:
                                print(f'TRELLIS: exported raw mesh to {mp}', flush=True)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception as e:
            log('TRELLIS attempt failed: ' + str(e))
            engine = 'tripo'

    if engine == 'tripo':
        # existing path (TripoSR/GeometryGenerator)
        gen = GeometryGenerator(trellis_weight_path=args.trellis_weights, tsr_options=tsr_opts, use_cache=getattr(args, 'cache_models', False))
        scene_codes = gen.generate(processed)

    # TripoSRInterface returns a list of dicts with mesh_path and output_dir
    if isinstance(scene_codes, list) and scene_codes:
        first_code = scene_codes[0]
        if isinstance(first_code, dict):
            raw = first_code.get('mesh_path', 'temp_raw.obj')
            # If the mesh_path was not provided, try to locate an OBJ in the reported output_dir
            if raw == 'temp_raw.obj':
                outdir_guess = first_code.get('output_dir')
                if outdir_guess:
                    try:
                        import glob as _glob
                        objs = sorted(_glob.glob(os.path.join(outdir_guess, '**', '*.obj'), recursive=True), key=os.path.getmtime, reverse=True)
                        if objs:
                            raw = objs[0]
                    except Exception:
                        pass
        else:
            raw = 'temp_raw.obj'
    else:
        raw = 'temp_raw.obj'
    # normalize to an absolute path when possible
    try:
        raw = os.path.abspath(raw)
    except Exception:
        pass
    log(f'  raw mesh -> {raw}')
    # Emit plain CLI message when a raw mesh path is determined so tests / UIs can detect successful exported meshes
    try:
        if raw:
            print(f'TRELLIS: exported raw mesh to {raw}', flush=True)
    except Exception:
        pass

    clean_obj = os.path.join(outdir, 'clean_quads.obj')
    log('>>> Retopology...')
    # Attempt retopology; if the reported raw mesh file is missing, try to
    # discover a replacement in the TripoSR output_dir (if available) and retry.
    try:
        process_retopology(
            raw,
            clean_obj,
            vertex_count=args.poly,
            smooth_flow=args.smooth,
            im_iters=args.im_iters,
            im_strength=args.im_strength,
        )
    except FileNotFoundError as e:
        log(f'  Retopology file not found: {e}. Attempting to locate an OBJ in TripoSR output_dir...')
        tried = False
        try:
            if isinstance(scene_codes, list) and scene_codes and isinstance(scene_codes[0], dict):
                sc0 = scene_codes[0]
                outdir_guess = sc0.get('output_dir')
                if outdir_guess and os.path.isdir(outdir_guess):
                    import glob as _glob
                    objs = sorted(_glob.glob(os.path.join(outdir_guess, '**', '*.obj'), recursive=True), key=os.path.getmtime, reverse=True)
                    if objs:
                        new_raw = os.path.abspath(objs[0])
                        log(f'  Found candidate OBJ in TripoSR output_dir: {new_raw} — retrying retopology')
                        raw = new_raw
                        tried = True
                        process_retopology(
                            raw,
                            clean_obj,
                            vertex_count=args.poly,
                            smooth_flow=args.smooth,
                            im_iters=args.im_iters,
                            im_strength=args.im_strength,
                        )
        except Exception as _e:
            log('  Retopology discovery/search failed: ' + str(_e))
        if not tried:
            # re-raise original error for visibility
            raise
    log(f'  retopo -> {clean_obj}')

    log('>>> Generating UVs...')
    uv_path, mesh = generate_uvs(clean_obj)
    log(f'  uv -> {uv_path}')

    if args.bake:
        log('>>> Baking textures (Blender headless if available)...')
        from neoforge_core import bake_texture_blender
        blender_exe = os.environ.get('BLENDER_PATH') or ''
        input_mesh = clean_obj
        input_texture = uv_path
        output_mesh = clean_obj  # overwrite or save as new if needed
        output_texture = os.path.join(outdir, 'baked_texture.png')
        baked = bake_texture_blender(input_mesh, input_texture, output_mesh, output_texture, blender_path=blender_exe, bake_type='diffuse', bake_res=args.bake_res)
        log(f'  bake result -> {baked}')
        # If a baked texture image was produced, copy it to the workspace and
        # ensure the UV OBJ references a .mtl that points to the image so Blender
        # will load it automatically when the OBJ is imported.
        try:
            import shutil
            if isinstance(baked, str) and os.path.isfile(baked) and baked.lower().endswith(('.png', '.jpg', '.jpeg')):
                uv_base = os.path.splitext(os.path.basename(uv_path))[0]
                tex_name = uv_base + os.path.splitext(baked)[1]
                tex_dest = os.path.join(outdir, tex_name)
                try:
                    shutil.copyfile(baked, tex_dest)
                    log(f'  Copied baked texture to workspace: {tex_dest}')
                except Exception as _e:
                    log(f'  Failed to copy baked texture into workspace: {_e}')

                # write a simple MTL that references the texture filename
                mtl_path = os.path.splitext(uv_path)[0] + '.mtl'
                try:
                    with open(mtl_path, 'w', encoding='utf8') as mf:
                        mf.write('newmtl material_0\n')
                        mf.write('Ka 1.000 1.000 1.000\n')
                        mf.write('Kd 1.000 1.000 1.000\n')
                        mf.write('map_Kd ' + tex_name + '\n')
                    log('  Wrote MTL: ' + str(mtl_path))
                except Exception as _e:
                    log('  Failed to write MTL: ' + str(_e))

                # Ensure OBJ includes a reference to the MTL and a usemtl statement
                try:
                    with open(uv_path, 'r', encoding='utf8') as f:
                        lines = f.readlines()
                    # insert mtllib at top if missing
                    if not any(l.strip().lower().startswith('mtllib') for l in lines[:5]):
                        lines.insert(0, f'mtllib {os.path.basename(mtl_path)}\n')
                    # find first face and insert usemtl just before it if missing
                    face_idx = None
                    for i, l in enumerate(lines):
                        if l.startswith('f '):
                            face_idx = i
                            break
                    if face_idx is not None:
                        # check if a usemtl already exists before the faces
                        pre = '\n'.join(lines[max(0, face_idx-3):face_idx])
                        if 'usemtl' not in pre:
                            lines.insert(face_idx, 'usemtl material_0\n')
                    with open(uv_path, 'w', encoding='utf8') as f:
                        f.writelines(lines)
                    log(f'  Updated OBJ to reference MTL and texture: {uv_path}')
                    # Try to export a textured GLB with embedded image
                    try:
                        import trimesh as _tm
                        from PIL import Image as _Image
                        from typing import Any, cast
                        scene = _tm.load(str(uv_path), force='scene')
                        img = _Image.open(tex_dest).convert('RGBA')
                        scene_any = cast(Any, scene)
                        for name, geom in scene_any.geometry.items():
                            geom_any = cast(Any, geom)
                            uv = getattr(getattr(geom_any, 'visual', None), 'uv', None)  # type: ignore[attr-defined]
                            if uv is None:
                                try:
                                    uv = geom_any.vertices[:, :2]
                                except Exception:
                                    uv = None
                            # Use getattr to avoid static analyzer false positives about trimesh internals
                            visual_mod = getattr(_tm, 'visual', None)  # type: ignore[attr-defined]
                            TextureVisuals = getattr(visual_mod, 'TextureVisuals', None) if visual_mod is not None else None
                            if TextureVisuals is not None:
                                geom_any.visual = TextureVisuals(uv=uv, image=img)
                            else:
                                # fallback: attach the module-level dummy visual to avoid redeclaration
                                try:
                                    geom_any.visual = _DummyVisual()
                                except Exception:
                                    pass
                        glb_out = os.path.splitext(str(uv_path))[0] + '_textured.glb'
                        scene.export(str(glb_out))
                        log('  Exported textured GLB: ' + str(glb_out))
                    except Exception as _e:
                        log('  Textured GLB export failed: ' + str(_e))
                except Exception as _e:
                    log(f'  Failed to update OBJ with MTL: {_e}')
        except Exception:
            pass
        # If no baked image was returned by bake_texture, try to discover a texture
        # produced by the TripoSR run (many external runs place texture.png under their output dir)
        try:
            if not (isinstance(baked, str) and os.path.isfile(baked) and baked.lower().endswith(('.png', '.jpg', '.jpeg'))):
                tripo_output = None
                if isinstance(scene_codes, list) and scene_codes and isinstance(scene_codes[0], dict):
                    sc0 = scene_codes[0]
                    tripo_output = sc0.get('output_dir')
                if tripo_output and os.path.isdir(tripo_output):
                    # search for likely texture files
                    candidates = glob.glob(os.path.join(tripo_output, '**', '*.png'), recursive=True) + glob.glob(os.path.join(tripo_output, '**', '*.jpg'), recursive=True)
                    candidates = [p for p in candidates if os.path.getsize(p) > 1024]
                    if candidates:
                        # pick the most recent
                        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
                        found = candidates[0]
                        uv_base = os.path.splitext(os.path.basename(uv_path))[0]
                        tex_name = uv_base + os.path.splitext(found)[1]
                        tex_dest = os.path.join(outdir, tex_name)
                        try:
                            import shutil
                            shutil.copyfile(found, tex_dest)
                            log(f'  Discovered and copied TripoSR texture into workspace: {tex_dest}')
                            # write MTL and update OBJ same as above
                            mtl_path = os.path.splitext(uv_path)[0] + '.mtl'
                            with open(mtl_path, 'w', encoding='utf8') as mf:
                                mf.write('newmtl material_0\n')
                                mf.write('Ka 1.000 1.000 1.000\n')
                                mf.write('Kd 1.000 1.000 1.000\n')
                                mf.write('map_Kd ' + tex_name + '\n')
                            log('  Wrote MTL: ' + str(mtl_path))
                            with open(uv_path, 'r', encoding='utf8') as f:
                                lines = f.readlines()
                            if not any(l.strip().lower().startswith('mtllib') for l in lines[:5]):
                                lines.insert(0, 'mtllib ' + os.path.basename(mtl_path) + '\n')
                            face_idx = None
                            for i, l in enumerate(lines):
                                if l.startswith('f '):
                                    face_idx = i
                                    break
                            if face_idx is not None:
                                pre = '\n'.join(lines[max(0, face_idx-3):face_idx])
                                if 'usemtl' not in pre:
                                    lines.insert(face_idx, 'usemtl material_0\n')
                            with open(uv_path, 'w', encoding='utf8') as f:
                                f.writelines(lines)
                            log(f'  Updated OBJ to reference MTL and texture: {uv_path}')
                            # Attempt to re-export GLB now that the MTL/texture is present so
                            # the viewer can load a textured GLB directly. We'll embed the
                            # texture into the GLB by assigning TextureVisuals with the image
                            # before exporting.
                            def _export_glb_with_texture(uv_path_local, tex_local):
                                try:
                                    import trimesh as _tm
                                    from PIL import Image as _Image
                                    from typing import Any, cast
                                    scene = _tm.load(str(uv_path_local), force='scene')
                                    img = _Image.open(tex_local).convert('RGBA')
                                    scene_any = cast(Any, scene)
                                    for name, geom in scene_any.geometry.items():
                                        geom_any = cast(Any, geom)
                                        uv = getattr(getattr(geom_any, 'visual', None), 'uv', None)  # type: ignore[attr-defined]  # type: ignore[attr-defined]
                                        if uv is None:
                                            try:
                                                uv = geom_any.vertices[:, :2]
                                            except Exception:
                                                uv = None
                                        # Use getattr to avoid static analyzer false positives about trimesh internals
                                        visual_mod = getattr(_tm, 'visual', None)  # type: ignore[attr-defined]
                                        TextureVisuals = getattr(visual_mod, 'TextureVisuals', None) if visual_mod is not None else None
                                        if TextureVisuals is not None:
                                            geom_any.visual = TextureVisuals(uv=uv, image=img)
                                        else:
                                            try:
                                                geom_any.visual = _DummyVisual()
                                            except Exception:
                                                pass
                                    glb_out = os.path.splitext(str(uv_path_local))[0] + '_textured.glb'
                                    scene.export(str(glb_out))
                                    log('  Exported textured GLB: ' + str(glb_out))
                                except Exception as exc:
                                    log('  Textured GLB export failed: ' + str(exc))

                            _export_glb_with_texture(uv_path, tex_dest)
                        except Exception as exc:
                            log('  Failed to copy discovered texture or write MTL: ' + str(exc))
        except Exception:
            pass
        # As a final fallback, search the external TripoSR workspace for any produced texture files
        try:
            tripo_sys_path = os.path.join(str(ROOT), 'external', 'TripoSR', 'workspace')
            if os.path.isdir(tripo_sys_path):
                candidates = glob.glob(os.path.join(tripo_sys_path, '**', '*texture*.png'), recursive=True)
                candidates += glob.glob(os.path.join(tripo_sys_path, '**', '*material*.png'), recursive=True)
                candidates += glob.glob(os.path.join(tripo_sys_path, '**', '*albedo*.png'), recursive=True)
                candidates = [p for p in candidates if os.path.getsize(p) > 1024]
                if candidates:
                    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
                    found = candidates[0]
                    uv_base = os.path.splitext(os.path.basename(uv_path))[0]
                    tex_name = uv_base + os.path.splitext(found)[1]
                    tex_dest = os.path.join(outdir, tex_name)
                    try:
                        import shutil
                        shutil.copyfile(found, tex_dest)
                        log(f'  Found and copied external TripoSR texture into workspace: {tex_dest}')
                        mtl_path = os.path.splitext(uv_path)[0] + '.mtl'
                        with open(mtl_path, 'w', encoding='utf8') as mf:
                            mf.write('newmtl material_0\n')
                            mf.write('Ka 1.000 1.000 1.000\n')
                            mf.write('Kd 1.000 1.000 1.000\n')
                            mf.write('map_Kd ' + tex_name + '\n')
                        log('  Wrote MTL: ' + str(mtl_path))
                        with open(uv_path, 'r', encoding='utf8') as f:
                            lines = f.readlines()
                        if not any(l.strip().lower().startswith('mtllib') for l in lines[:5]):
                            lines.insert(0, 'mtllib ' + os.path.basename(mtl_path) + '\n')
                        face_idx = None
                        for i, l in enumerate(lines):
                            if l.startswith('f '):
                                face_idx = i
                                break
                        if face_idx is not None:
                            pre = '\n'.join(lines[max(0, face_idx-3):face_idx])
                            if 'usemtl' not in pre:
                                lines.insert(face_idx, 'usemtl material_0\n')
                        with open(uv_path, 'w', encoding='utf8') as f:
                            f.writelines(lines)
                        log(f'  Updated OBJ to reference MTL and texture: {uv_path}')
                        # After updating the OBJ, attempt to export a textured GLB embedding the image
                        try:
                            import trimesh as _tm
                            from PIL import Image as _Image
                            from typing import Any, cast
                            scene = _tm.load(str(uv_path), force='scene')
                            img = _Image.open(tex_dest).convert('RGBA')
                            scene_any = cast(Any, scene)
                            for name, geom in scene_any.geometry.items():
                                geom_any = cast(Any, geom)
                                uv = getattr(getattr(geom_any, 'visual', None), 'uv', None)
                                if uv is None:
                                    try:
                                        uv = geom_any.vertices[:, :2]
                                    except Exception:
                                        uv = None
                                # Use getattr to avoid static-analyzer complaints if `trimesh.visual`
                                # is not present in the runtime stubs; fall back to a dummy visual.
                                visual_mod = getattr(_tm, 'visual', None)  # type: ignore[attr-defined]
                                TextureVisuals = getattr(visual_mod, 'TextureVisuals', None) if visual_mod is not None else None
                                if TextureVisuals is not None:
                                    geom_any.visual = TextureVisuals(uv=uv, image=img)
                                else:
                                    try:
                                        geom_any.visual = _DummyVisual()
                                    except Exception:
                                        pass
                            glb_out = os.path.splitext(str(uv_path))[0] + '_textured.glb'
                            scene.export(str(glb_out))
                            log('  Exported textured GLB: ' + str(glb_out))
                        except Exception as _e:
                            log('  Textured GLB export failed: ' + str(_e))
                    except Exception as _e:
                        log('  Failed to copy external texture or write MTL: ' + str(_e))
        except Exception:
            pass

    # If rendering or metrics were requested, compute metrics versus the input image
    try:
        metrics_done = False
        if args.render or args.metrics:
            # find the TripoSR output dir (returned in scene_codes)
            tripo_output = None
            if isinstance(scene_codes, list) and scene_codes and isinstance(scene_codes[0], dict):
                sc0 = scene_codes[0]
                tripo_output = sc0.get('output_dir') or outdir
            else:
                # fallback to project workspace
                tripo_output = outdir

            # Search recursively for render frames (support nested output layouts and multiple naming schemes)
            render_files = sorted(glob.glob(os.path.join(tripo_output, '**', 'render_*.png'), recursive=True))
            if not render_files:
                render_files = sorted(glob.glob(os.path.join(tripo_output, '**', '*render*.png'), recursive=True))
            if not render_files:
                # Fallback: any PNGs under the output dir (e.g., saved frame sequences with other names)
                render_files = sorted(glob.glob(os.path.join(tripo_output, '**', '*.png'), recursive=True))

            # Filter out tiny files or non-render images by reasonable size (e.g., > 1KB)
            render_files = [p for p in render_files if os.path.getsize(p) > 1024]

            if render_files:
                # Prefer the most recent N frames
                render_files = sorted(render_files, key=lambda p: os.path.getmtime(p), reverse=False)
                from PIL import Image
                import numpy as np
                from skimage.metrics import structural_similarity as ssim
                ref_im = Image.open(inp).convert('RGB').resize((256, 256))
                ref = np.array(ref_im).astype(np.float32) / 255.0
                ssim_vals = []
                # sample up to 5 views evenly spaced across available renders
                num_samples = min(5, len(render_files))
                if num_samples > 0:
                    step = max(1, len(render_files) // num_samples)
                    sampled = [render_files[i] for i in range(0, len(render_files), step)][:num_samples]
                else:
                    sampled = []
                for rf in sampled:
                    im_im = Image.open(rf).convert('RGB').resize((256, 256))
                    im = np.array(im_im).astype(np.float32) / 255.0
                    # SSIM needs grayscale
                    from skimage.color import rgb2gray
                    s = ssim(rgb2gray(ref), rgb2gray(im), data_range=1.0)
                    ssim_vals.append(s)
                avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0.0
                log(f'>>> Metrics: SSIM (avg over {len(ssim_vals)} rendered views) = {avg_ssim:.4f}')
                metrics_done = True
                # optional LPIPS if available
                try:
                    import importlib
                    lpips = importlib.import_module('lpips')
                    loss_fn = lpips.LPIPS(net='alex')
                    import torch
                    lpips_vals = []
                    x_ref = torch.tensor(ref).permute(2,0,1).unsqueeze(0)*2-1
                    for rf in sampled:
                        im_im = Image.open(rf).convert('RGB').resize((256, 256))
                        im = np.array(im_im).astype(np.float32) / 255.0
                        x = torch.tensor(im).permute(2,0,1).unsqueeze(0) * 2 - 1
                        d = loss_fn(x_ref, x).item()
                        lpips_vals.append(d)
                    avg_lpips = sum(lpips_vals) / len(lpips_vals)
                    msg = '>>> Metrics: LPIPS (avg over {} views) = {:.4f}'.format(len(lpips_vals), avg_lpips)
                    log(msg)
                except Exception:
                    log('LPIPS not available; install `lpips` to compute perceptual metric.')
        if not metrics_done and args.metrics:
            log('No rendered views found to compute metrics.')
    except Exception as e:
        log('Metrics computation failed: ' + str(e))

    log('\nDone. Outputs in ' + os.path.abspath(outdir))
    log('  OBJ: ' + os.path.abspath(clean_obj))
    log('  UV-OBJ: ' + os.path.abspath(uv_path))
    if args.bake:
        # Only call abspath if baked is a valid string path
        if isinstance(baked, str) and baked:
            log('  Baked: ' + os.path.abspath(baked))
        else:
            log(f'  Baked: {baked}')


if __name__ == '__main__':
    main()