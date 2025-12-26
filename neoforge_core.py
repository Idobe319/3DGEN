from typing import Optional, Any, Tuple, cast, Callable, ContextManager

def bake_texture_blender(input_mesh: str, input_texture: str, output_mesh: str, output_texture: str, blender_path: Optional[str] = None, bake_type: str = 'diffuse', bake_res: int = 2048) -> bool:
    """
    מבצע baking של טקסטורה על רשת retopo באמצעות Blender headless.
    דורש התקנת Blender והגדרת הנתיב ל-blender.exe.
    """
    import shutil
    import tempfile
    import subprocess
    import os
    blender_exe = blender_path or shutil.which('blender')
    if not blender_exe or not os.path.exists(blender_exe):
        nlog(f"Blender executable not found: {blender_exe}")
        return False
    # יצירת סקריפט Python זמני ל-Blender
    script = f"""
import bpy
bpy.ops.wm.open_mainfile(filepath=r'{input_mesh}')
# יצירת חומרים וטקסטורה
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        mat = bpy.data.materials.new(name='BakedMaterial')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(r'{input_texture}')
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
        obj.data.materials.append(mat)
        # יצירת תמונה חדשה ל-bake
        bake_img = bpy.data.images.new('BakeResult', width={bake_res}, height={bake_res})
        bake_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        bake_node.image = bake_img
        mat.node_tree.links.new(bsdf.inputs['Base Color'], bake_node.outputs['Color'])
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.bake(type='{bake_type.upper()}', use_clear=True, margin=2)
        bake_img.save_render(r'{output_texture}')
bpy.ops.wm.save_as_mainfile(filepath=r'{output_mesh}')
"""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        cmd = [blender_exe, '--background', '--python', script_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(output_texture):
            nlog(f"Blender baking completed: {output_texture}")
            return True
        else:
            nlog(f"Blender baking failed: {result.stderr.decode()}")
            return False
    finally:
        os.remove(script_path)
import os
import io
import subprocess
from typing import Any, Tuple, cast, Callable, ContextManager
from pathlib import Path

import torch
import numpy as np
import trimesh
from trimesh import Trimesh
from trimesh.visual import TextureVisuals
# Optional trimesh smoothing utilities (Taubin smoothing)
try:
    import trimesh.smoothing  # type: ignore
except Exception:
    trimesh.smoothing = None  # type: ignore
from PIL import Image
import shutil  # used to relocate TripoSR-generated meshes to expected paths
from threading import RLock
from typing import Callable, Dict, Tuple, Any


# --- Simple thread-safe in-memory model cache ---
class ModelCache:
    """Thread-safe cache for large model objects (TRELLIS / TripoSR wrappers).

    Usage: ModelCache.get(key, loader_fn) will call loader_fn() on first access
    and reuse the result on subsequent calls.
    """

    def __init__(self):
        self._lock = RLock()
        self._store: Dict[Tuple[str, str], Any] = {}

    def get(self, engine: str, identifier: str, loader: Callable[[], Any]) -> Any:
        key = (engine, identifier or "__default__")
        with self._lock:
            if key in self._store:
                return self._store[key]
            val = loader()
            self._store[key] = val
            return val

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Global cache instance used across the module
_model_cache = ModelCache()

# optional logger: fall back to print if not present
try:
    from neoforge_logger import log as nlog
except Exception:
    nlog = print

def _gpu_stats() -> dict | None:
    """Return GPU telemetry from best available source.

    Priority:
    1. pynvml (NVML) if available
    2. nvidia-smi subprocess parse
    3. torch.cuda (best-effort)
    Returns a dict containing memory_used/memory_total (bytes), utilization (percent) and device name where available.
    """
    try:
        # 1) Try NVML via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            name = pynvml.nvmlDeviceGetName(handle)
            try:
                device = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            except Exception:
                device = str(name)
            return {
                'memory_used': int(mem.used),
                'memory_total': int(mem.total),
                'utilization': int(util),
                'device': device,
            }
        except Exception:
            pass

        # 2) Fallback to nvidia-smi parsing (returns MiB values)
        try:
            import subprocess
            out = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
                '-i', '0'
            ], text=True).strip()
            if out:
                parts = [p.strip() for p in out.split(',')]
                if len(parts) >= 3:
                    used_mib = int(parts[0])
                    total_mib = int(parts[1])
                    util = int(parts[2])
                    return {
                        'memory_used': used_mib * 1024 * 1024,
                        'memory_total': total_mib * 1024 * 1024,
                        'utilization': util,
                        'device': 'NVIDIA',
                    }
        except Exception:
            pass

        # 3) Fallback to torch (process-local information)
        try:
            if torch.cuda.is_available():
                return {
                    'memory_allocated': int(torch.cuda.memory_allocated()),
                    'memory_reserved': int(torch.cuda.memory_reserved()),
                    'device': torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'cuda',
                }
        except Exception:
            pass
    except Exception:
        pass
    return None


def sanitize_for_trellis(img, bg=(255, 255, 255), min_nonzero_alpha=10000, min_size=512, white_bg=True, white_thresh=30):
    """Ensure TRELLIS receives a valid image with a non-empty alpha mask.

    This function mirrors the logic used in `run_local._run_trellis` and is
    exposed here so it can be tested directly. It returns a PIL RGBA image
    with a non-empty alpha channel and ensures a minimum size.
    """
    from PIL import Image as _PILImage
    import numpy as _np

    # Coerce non-PIL inputs into a PIL Image
    if not isinstance(img, _PILImage.Image):
        try:
            img = _PILImage.fromarray(_np.array(img))
        except Exception:
            raise RuntimeError('sanitize_for_trellis: unable to coerce input to PIL Image')

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
                nlog(f"TRELLIS sanitize: heuristic mask produced empty result; using full-image mask")
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
                    nlog(f"TRELLIS sanitize: existing alpha had only {nonzero} nonzero pixels; replaced with heuristic mask ({int(mask.sum())} nonzero)" )
                except Exception:
                    print(f"TRELLIS sanitize: existing alpha had only {nonzero} nonzero pixels; replaced with heuristic mask ({int(mask.sum())} nonzero)")
            else:
                # fallback to full mask
                alpha = _np.ones((h, w), dtype=_np.uint8) * 255
                try:
                    nlog(f"TRELLIS sanitize: existing alpha nearly empty and heuristic empty; falling back to full mask")
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
        nlog(f"TRELLIS sanitize result: mode=RGBA, size={img_rgba.size}, alpha_min={a_min}, alpha_max={a_max}, alpha_mean={a_mean:.2f}, alpha_nonzero={nonzero}, bbox={bbox}")
    except Exception:
        print(f"TRELLIS sanitize result: mode=RGBA, size={img_rgba.size}, alpha_min={a_min}, alpha_max={a_max}, alpha_mean={a_mean:.2f}, alpha_nonzero={nonzero}, bbox={bbox}")

    return img_rgba


def preprocess_image(input_path) -> str:
    nlog(">>> Stage 1: Preprocessing & Background Removal...")
    nlog.progress('preprocess', percent=0, details={'phase': 'start', 'gpu': _gpu_stats()})
    img = Image.open(input_path).convert("RGBA")
    # Try using rembg (may use ONNX GPU provider). Import lazily to avoid
    # loading ONNX/CUDA providers at module import time. If it fails, fall
    # back to a simple CPU-based background removal using a color threshold.
    img_clean = None
    try:
        import importlib
        rembg_mod = importlib.import_module('rembg')
        remove = getattr(rembg_mod, 'remove')
        img_clean_data: Any = remove(img)
        if isinstance(img_clean_data, bytes):
            img_clean = Image.open(io.BytesIO(img_clean_data)).convert("RGBA")
        elif isinstance(img_clean_data, np.ndarray):
            img_clean = Image.fromarray(img_clean_data).convert("RGBA")
        else:
            img_clean = img_clean_data
    except Exception as e:
        nlog(f"rembg failed or unavailable ({e}), using CPU fallback removal. To enable improved background removal, install rembg: pip install rembg")
        # Simple fallback: create alpha mask by thresholding near-white background
        rgba = img.copy()
        pixels = np.array(rgba)
        # Detect non-white pixels (tolerance)
        tol = 20
        # Ensure arithmetic is performed with a signed integer type so
        # subtraction with 255 and abs() behave as expected and static
        # type-checkers don't complain about operations with Python literals.
        rgb = pixels[..., :3].astype(np.int16)
        mask = ~np.all(np.abs(rgb - 255) <= tol, axis=-1)
        # if image has transparent pixels already, respect them
        if pixels.shape[-1] == 4:
            alpha = pixels[..., 3] > 0
            mask = mask | alpha
        # Build PIL image from mask
        mask_img = Image.fromarray((mask * 255).astype('uint8'), mode='L')
        img_clean = Image.new('RGBA', img.size, (255,255,255,0))
        img_clean.paste(img, mask=mask_img)
    bbox = img_clean.getbbox()
    if bbox:
        img_clean = img_clean.crop(bbox)
    processed_path = "temp_input.png"
    img_clean.save(processed_path)
    nlog.progress('preprocess', percent=100, details={'phase': 'done', 'gpu': _gpu_stats()})
    return processed_path

# --- מודול 2: יצירה גולמית (The Generator) ---
# Single `GeometryGenerator` class that attempts to load TRELLIS/TripoSR
# at runtime and falls back to a procedural sphere generator when
# the model or weights are not available. This avoids redeclaring the
# class name twice (which Pylance warns about).
class GeometryGenerator:
    from typing import Optional

    def __init__(self, trellis_weight_path: Optional[str] = None, tsr_options: dict | None = None, use_cache: bool = False):
        print(">>> Initializing GeometryGenerator...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tsr_options = tsr_options or {}
        try:
            from tsr.system import TSR  # may be a local stub or real model
            nlog(">>> Loading AI Geometry Engine (TRELLIS)...")
            nlog.progress('model_load', percent=0, details={'phase': 'start', 'gpu': _gpu_stats()})

            def _load_tsr():
                # allow passing a direct weight path
                if trellis_weight_path is not None:
                    return TSR.from_pretrained("stabilityai/TripoSR", weight_path=trellis_weight_path)
                else:
                    return TSR.from_pretrained("stabilityai/TripoSR")

            # Use the shared cache if requested
            try:
                if use_cache:
                    ident = str(trellis_weight_path) if trellis_weight_path is not None else "stabilityai/TripoSR"
                    self.model = _model_cache.get('tripo', ident, _load_tsr)
                else:
                    self.model = _load_tsr()
            except Exception:
                # fallback to direct load if cache attempt fails
                try:
                    self.model = _load_tsr()
                except Exception:
                    raise

            # Measure and report load time
            try:
                to_fn = getattr(self.model, "to", None)
                if callable(to_fn):
                    to_fn(self.device)
                else:
                    to_dev = getattr(self.model, "to_device", None)
                    if callable(to_dev):
                        to_dev(self.device)
                    else:
                        nlog("Model does not support .to()/to_device(); skipping device placement.")
            except Exception:
                pass

            # attempt a *safe* warm-up using model-provided hooks when available
            try:
                nlog("   - Warm-up: attempting a light model initialization without invoking full export/run")
                nlog.progress('model_load', percent=50, details={'phase': 'warmup', 'gpu': _gpu_stats()})
                warmed = False
                try:
                    # Prefer a dedicated warmup() hook if the model exposes one
                    warm_fn = getattr(self.model, 'warmup', None)
                    if callable(warm_fn):
                        try:
                            warm_fn(device=self.device)
                            warmed = True
                        except Exception:
                            warmed = False
                except Exception:
                    warmed = False

                # If no hook, try calling a light-weight encoder/backbone if present
                if not warmed:
                    try:
                        enc = getattr(self.model, 'encoder', None) or getattr(self.model, 'backbone', None)
                        if enc is not None and callable(enc):
                            # construct a small random tensor compatible with common vision backbones
                            with torch.no_grad():
                                dummy = torch.zeros((1, 3, 32, 32), device=(self.device if self.device == 'cuda' else 'cpu'))
                                try:
                                    _ = enc(dummy)
                                    warmed = True
                                except Exception:
                                    warmed = False
                    except Exception:
                        warmed = False

                if not warmed:
                    nlog('   - No lightweight warmup available; skipping dummy model call to avoid duplicate heavy runs')
                nlog.progress('model_load', percent=100, details={'phase': 'done', 'gpu': _gpu_stats()})
            except Exception:
                nlog.progress('model_load', percent=100, details={'phase': 'done', 'gpu': _gpu_stats()})
        except Exception:
            nlog("Warning: TRELLIS/TripoSR not available — using fallback generator.")
            nlog.progress('model_load', percent=100, details={'phase': 'unavailable', 'gpu': _gpu_stats()})

    def generate(self, image_path) -> str:
        """Generate a mesh from an input image and return the mesh file path.

        This normalizes the various possible return formats from the TSR wrapper
        (a list of dicts, a dict, or a direct path string) and guarantees a
        single path string is returned for downstream stages and tests.
        """
        if self.model is not None:
            nlog(">>> Stage 2: Generating Raw Geometry (TRELLIS)...")
            nlog.progress('inference', percent=0, details={'phase': 'start', 'gpu': _gpu_stats()})
            img = Image.open(image_path)
            with torch.no_grad():
                # Use autocast when running on CUDA for faster/more memory-efficient inference
                from contextlib import nullcontext
                # Choose an appropriate autocast context that exists in the
                # current PyTorch build. Prefer torch.cuda.amp.autocast when
                # running on CUDA, otherwise fall back to torch.amp.autocast
                # if available; finally use a nullcontext as a no-op.
                # Type-annotate `ac` so static analyzers know it is a callable returning a context-manager
                ac: Callable[[], ContextManager[Any]]
                # Prefer the public `torch.amp.autocast(device_type='cuda')` API when running on CUDA
                ac_fn = getattr(torch, 'amp', None)
                if (self.device == 'cuda' and torch.cuda.is_available() and ac_fn is not None and hasattr(ac_fn, 'autocast')):
                    try:
                        ac = lambda: cast(ContextManager[Any], ac_fn.autocast(device_type='cuda'))
                    except Exception:
                        ac = nullcontext
                else:
                    ac = nullcontext
                with ac():
                    # Pass tsr-specific options (mc_resolution, chunk_size, bake_texture, texture_resolution, render)
                    scene_codes = self.model([img], device=self.device, **self.tsr_options)
            nlog.progress('inference', percent=100, details={'phase': 'done', 'gpu': _gpu_stats()})

            # Normalize common return types to a single mesh path string
            if isinstance(scene_codes, (list, tuple)) and scene_codes:
                first = scene_codes[0]
                if isinstance(first, dict) and 'mesh_path' in first:
                    mesh_path = first['mesh_path']
                    # Log the reported mesh_path immediately for diagnostics
                    try:
                        nlog(f"TSR reported mesh_path: {mesh_path}")
                    except Exception:
                        pass
                    # If TripoSR produced a mesh, copy it to the runner's expected
                    # temporary filename so downstream stages (Instant Meshes, etc.)
                    # receive a consistent path (temp_raw.obj).
                    try:
                        if os.path.exists(mesh_path):
                            dest = os.path.abspath('temp_raw.obj')
                            try:
                                shutil.copyfile(mesh_path, dest)
                                nlog(f"TSR: copied mesh {mesh_path} -> {dest}")
                                return dest
                            except Exception as _e:
                                nlog(f"TSR: failed to copy mesh: {_e}; returning original path {mesh_path}")
                                return mesh_path
                        else:
                            # If the reported mesh path doesn't exist, try searching the
                            # reported output_dir for any .obj files and copy the newest one.
                            outdir = first.get('output_dir')
                            try:
                                if outdir:
                                    nlog(f"TSR reported output_dir: {outdir}")
                            except Exception:
                                pass
                            if outdir and os.path.isdir(outdir):
                                import glob as _glob
                                objs = sorted(_glob.glob(os.path.join(outdir, '**', '*.obj'), recursive=True), key=os.path.getmtime, reverse=True)
                                if objs:
                                    found = objs[0]
                                    dest = os.path.abspath('temp_raw.obj')
                                    try:
                                        shutil.copyfile(found, dest)
                                        nlog(f"TSR: found mesh in output_dir and copied {found} -> {dest}")
                                        return dest
                                    except Exception as _e:
                                        nlog(f"TSR: failed to copy discovered mesh: {_e}; returning original path {mesh_path}")
                                        return mesh_path
                            # fallback: return the original (may be handled by caller)
                            return mesh_path
                    except Exception:
                        return mesh_path
                if isinstance(first, str):
                    return first
            if isinstance(scene_codes, dict) and 'mesh_path' in scene_codes:
                return cast(dict, scene_codes)['mesh_path']
            if isinstance(scene_codes, str):
                return scene_codes

            # Extra check: TripoSR sometimes writes to the canonical external workspace
            # path; check there for 'mesh.obj' and copy it to temp_raw.obj if present.
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'TripoSR', 'workspace', 'triposr_run', '0'))
                expected_output = os.path.join(base_dir, 'mesh.obj')
                if os.path.exists(expected_output):
                    dest = os.path.abspath('temp_raw.obj')
                    try:
                        shutil.copyfile(expected_output, dest)
                        nlog(f"TSR: found canonical mesh {expected_output} and copied to {dest}")
                        return dest
                    except Exception as _e:
                        nlog(f"TSR: failed to copy canonical mesh: {_e}")
            except Exception:
                pass

            # If we reach here, we couldn't find a suitable mesh path
            raise FileNotFoundError(f"TripoSR completed but no OBJ mesh was discovered. scene_codes: {scene_codes}")
        else:
            nlog(">>> Stage 2: Generating Raw Geometry (Fallback sphere)...")
            sphere = trimesh.primitives.Sphere(radius=1.0, subdivisions=4)
            raw_path = "temp_raw.obj"
            sphere.export(raw_path)
            return raw_path

def process_retopology(input_obj: Any, output_obj: str, vertex_count: int = 2000, smooth_flow: bool = False, im_iters: int = 0, im_strength: float = 0.5) -> str:
    nlog(f">>> Stage 3: Internal Retopology (Target: {vertex_count} verts)...")
    nlog.progress('retopology', percent=0, details={'phase': 'start', 'target_vertices': vertex_count, 'gpu': _gpu_stats()})
    # Normalize input: allow passing the TripoSR scene_codes/list/dict directly
    try:
        # If a TripoSR scene_codes list/dict was passed, extract the mesh path
        if isinstance(input_obj, (list, tuple)) and input_obj:
            first = input_obj[0]
            if isinstance(first, dict) and 'mesh_path' in first:
                input_obj = first['mesh_path']
            elif isinstance(first, str):
                input_obj = first
        elif isinstance(input_obj, dict) and 'mesh_path' in input_obj:
            input_obj = input_obj['mesh_path']
    except Exception:
        pass

    # Ensure we have a path-like string for subprocess and trimesh.load
    if not isinstance(input_obj, (str, bytes, os.PathLike)):
        try:
            input_obj = str(input_obj)
        except Exception:
            raise TypeError("input_obj must be a path-like object or contain 'mesh_path'") from None

    # Prefer using the Instant Meshes binary when available for higher-quality
    # field-aligned quad remeshing. If it fails or isn't present, fall back to
    # the internal pure-Python retopology implemented below.
    try:
        possible = [
            os.path.join(os.path.dirname(__file__), "Instant Meshes.exe"),
            os.path.join(os.path.dirname(__file__), "InstantMeshes.exe"),
            os.path.join(os.path.dirname(__file__), "instant-meshes.exe"),
        ]
        exe = next((p for p in possible if os.path.exists(p)), None)
        if exe:
            print(f">>> Found Instant Meshes executable at {exe}, running it for retopology...")
            nlog.progress('retopology', percent=30, details={'phase': 'instant_meshes', 'gpu': _gpu_stats()})
            try:
                # Instant Meshes CLI: try the common '-i <in>' style first, then
                # fallback to '<in> -o <out>' if the first form fails on some builds.
                tried_ok = False
                try:
                    subprocess.run([exe, "-i", input_obj, "-o", output_obj], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    tried_ok = True
                except Exception:
                    try:
                        subprocess.run([exe, input_obj, "-o", output_obj], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        tried_ok = True
                    except Exception:
                        tried_ok = False

                if not tried_ok:
                    raise RuntimeError('Instant Meshes invocation failed for both argument styles')

                nlog(f">>> Instant Meshes produced: {output_obj}")
                nlog.progress('retopology', percent=80, details={'phase': 'instant_meshes_exported', 'gpu': _gpu_stats()})
                # if smoothing requested, apply a light Laplacian smoothing pass
                try:
                    m = cast(Trimesh, trimesh.load(output_obj, process=False))
                    # build adjacency
                    nbrs = [[] for _ in range(len(m.vertices))]
                    for f in m.faces:
                        nbrs[f[0]].extend([f[1], f[2]])
                        nbrs[f[1]].extend([f[0], f[2]])
                        nbrs[f[2]].extend([f[0], f[1]])

                    # Base smoothing iterations influenced by `smooth_flow` boolean
                    base_iters = 2 if smooth_flow else 0
                    total_iters = base_iters + max(0, int(im_iters))
                    weight = float(im_strength)
                    for _ in range(total_iters):
                        newv = m.vertices.copy()
                        for vi, neigh in enumerate(nbrs):
                            if not neigh:
                                continue
                            nb = list(set(neigh))
                            avg = m.vertices[nb].mean(axis=0)
                            newv[vi] = (1.0 - weight) * newv[vi] + weight * avg
                        m.vertices = newv
                    m.export(output_obj)
                except Exception:
                    pass
                return output_obj
            except Exception as e:
                nlog(f"Instant Meshes failed ({e}), falling back to internal retopology.")
    except Exception:
        pass
    # Pure-Python retopology implementation — voxel clustering for vertex
    # reduction followed by triangle-pairing to form quad faces when possible.
    # Ensure the loader receives a string path to satisfy static typing checks
    input_path = str(input_obj)
    # If the path does not exist, attempt to locate the file by basename
    if not os.path.exists(input_path):
        try:
            basename = os.path.basename(input_path)
            # search current working dir and project tree for a matching file
            cand = None
            for root, _dirs, files in os.walk(os.getcwd()):
                if basename in files:
                    cand = os.path.join(root, basename)
                    break
            if cand:
                nlog(f"Input mesh '{input_path}' not found; using discovered file: {cand}")
                input_path = cand
            else:
                # try script directory
                script_dir = os.path.dirname(__file__)
                alt = os.path.join(script_dir, basename)
                if os.path.exists(alt):
                    nlog(f"Input mesh '{input_path}' not found; using file in script dir: {alt}")
                    input_path = alt
        except Exception:
            pass

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Retopology input file not found: {input_path}")

    mesh = cast(Trimesh, trimesh.load(input_path, process=False))

    # If the input is a point cloud with no faces, attempt surface reconstruction
    # using Open3D (Poisson or Ball Pivoting), then simplify to the requested
    # vertex count. This ensures we produce a connected, smooth mesh instead of
    # leaving downstream stages with unconnected points.
    try:
        has_faces = bool(getattr(mesh, 'faces', None) is not None and len(mesh.faces) > 0)
    except Exception:
        has_faces = False

    if not has_faces:
        try:
            import open3d as o3d  # type: ignore[reportMissingImports]
            # Cast the imported Open3D module to Any so static analyzers (Pylance/pyright)
            # don't report attribute access errors for runtime-provided submodules.
            from typing import Any
            o3d = cast(Any, o3d)

            nlog('   - Input appears to be a point cloud (no faces). Attempting surface reconstruction with Open3D...')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

            # Estimate normals (required for Poisson reconstruction)
            try:
                # use a heuristic radius based on bounding box size
                bbox = np.asarray(mesh.bounds)
                diag = float(np.linalg.norm(bbox[1] - bbox[0]))
                radius = max(0.01, diag / 50.0)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2.0, max_nn=30))
                pcd.normalize_normals()
            except Exception:
                try:
                    pcd.estimate_normals()
                except Exception:
                    pass

            # Poisson reconstruction (depth tuned toward requested vertex count)
            try:
                depth = int(np.clip(max(7, min(11, int(np.log2(max(vertex_count, 512))))), 6, 12))
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
                densities = np.asarray(densities)

                # Remove very low-density vertices to discard spurious islands
                try:
                    thresh = np.quantile(densities, 0.02)
                    vert_mask = densities > thresh
                    if vert_mask.sum() > 4:
                        faces_o3d = np.asarray(mesh_o3d.triangles)
                        verts_o3d = np.asarray(mesh_o3d.vertices)
                        keep_face_mask = np.all(vert_mask[faces_o3d], axis=1)
                        faces_o3d = faces_o3d[keep_face_mask]
                        mesh_trimesh = trimesh.Trimesh(vertices=verts_o3d, faces=faces_o3d, process=True)
                    else:
                        raise RuntimeError('Poisson produced too few dense vertices')
                except Exception:
                    # If density filtering fails, accept raw Poisson mesh
                    mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles), process=True)
            except Exception as e_poisson:
                nlog(f'   - Poisson reconstruction failed ({e_poisson}); trying Ball-Pivoting fallback...')
                try:
                    distances = pcd.compute_nearest_neighbor_distance()
                    avg_dist = float(np.mean(distances)) if len(distances) else 0.01
                    radii = o3d.utility.DoubleVector([avg_dist * s for s in (1.5, 3.0, 6.0)])
                    bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
                    mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(bmesh.vertices), faces=np.asarray(bmesh.triangles), process=True)
                except Exception as e_bp:
                    nlog(f'   - Ball Pivoting fallback failed ({e_bp}) — cannot reconstruct; falling back to voxel clustering')
                    mesh_trimesh = None

            # Simplify the reconstructed mesh to the target vertex budget if needed
            if 'mesh_trimesh' in locals() and mesh_trimesh is not None:
                try:
                    cur_v = len(mesh_trimesh.vertices)
                    if vertex_count and cur_v > vertex_count:
                        try:
                            mo = o3d.geometry.TriangleMesh()
                            mo.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
                            mo.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
                            target_tri = max(100, int(vertex_count * 2))
                            mo = mo.simplify_quadric_decimation(target_tri)
                            mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mo.vertices), faces=np.asarray(mo.triangles), process=True)
                        except Exception:
                            pass

                except Exception:
                    pass

                try:
                    mesh_trimesh.export(output_obj)
                    nlog(f"   - Reconstructed mesh from point cloud and exported to {output_obj}")
                    return output_obj
                except Exception as _e:
                    nlog(f'   - Failed to export reconstructed mesh: {_e}')
        except Exception as _e:
            nlog(f'   - Point cloud reconstruction not available: {_e}; continuing with internal retopology')

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # If target vertex count is similar or larger than current, skip reduction
    if vertex_count < len(verts):
        # compute voxel size to reduce to approx vertex_count
        min_b = verts.min(axis=0)
        max_b = verts.max(axis=0)
        bbox_size = max_b - min_b
        vol = max(bbox_size[0] * bbox_size[1] * bbox_size[2], 1e-9)
        voxel_size = (vol / max(vertex_count, 1)) ** (1.0 / 3.0)

        # assign vertices to voxels
        voxel_idx = ((verts - min_b) / (voxel_size + 1e-12)).astype(int)
        vox_map = {}
        for i, v in enumerate(verts):
            key = tuple(voxel_idx[i])
            vox_map.setdefault(key, []).append(i)

        new_verts = []
        old_to_new = np.full(len(verts), -1, dtype=int)
        for new_i, (k, idcs) in enumerate(vox_map.items()):
            pts = verts[idcs]
            new_verts.append(pts.mean(axis=0))
            for oi in idcs:
                old_to_new[oi] = new_i

        new_verts = np.array(new_verts)

        # rebuild faces with new indices, discard degenerate faces
        new_faces = []
        for f in faces:
            nf = old_to_new[f]
            if len(set(nf)) == 3:
                new_faces.append(nf)
        faces = np.array(new_faces, dtype=int)
        verts = new_verts

    # Attempt to pair adjacent triangles into quads
    # build edge->faces map
    edge_map = {}
    for fi, f in enumerate(faces):
        for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            key = tuple(sorted(e))
            edge_map.setdefault(key, []).append(int(fi))

    used_tri = [False] * len(faces)
    quads = []
    for edge, fids in edge_map.items():
        if len(fids) != 2:
            continue
        a, b = int(fids[0]), int(fids[1])
        # debug: ensure indices are ints
        if not isinstance(a, int) or not isinstance(b, int):
            print('DEBUG: non-int indices in edge_map', type(a), type(b), fids)
        if used_tri[a] or used_tri[b]:
            continue
        fa = faces[a]
        fb = faces[b]
        verts_comb = list(set(tuple(fa)) | set(tuple(fb)))
        if len(verts_comb) != 4:
            continue
        # check planarity: angle between face normals
        na = trimesh.triangles.normals(verts[fa][None, :, :])[0].ravel()
        nb = trimesh.triangles.normals(verts[fb][None, :, :])[0].ravel()
        dot = float(np.dot(na, nb))
        if np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) > 20.0:
            continue
        # order quad vertices consistently (project to plane)
        quad_vs = verts[verts_comb]
        center = quad_vs.mean(axis=0)
        # compute a best-fit plane normal
        normal = np.cross(quad_vs[1] - quad_vs[0], quad_vs[2] - quad_vs[0])
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)
        # compute angles around center
        ref = quad_vs[0] - center
        ref = ref / (np.linalg.norm(ref) + 1e-12)
        angles = []
        for v in quad_vs:
            vec = v - center
            # project into plane
            proj = vec - normal * np.dot(vec, normal)
            ang = np.arctan2(np.dot(np.cross(ref, proj), normal), np.dot(ref, proj))
            angles.append(ang)
        order = np.argsort(angles)
        ordered = [int(verts_comb[i]) for i in order]
        quads.append(ordered)
        used_tri[int(a)] = True
        used_tri[int(b)] = True

    # collect leftover triangles
    triangles = [list(faces[i]) for i in range(len(faces)) if not used_tri[i]]

    # write OBJ with quads and triangles
    def write_obj_with_quads(path, verts, quads, triangles):
        with open(path, 'w', encoding='utf8') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            f.write("g retopo\n")
            for q in quads:
                f.write("f " + " ".join(str(i+1) for i in q) + "\n")
            for t in triangles:
                f.write("f " + " ".join(str(i+1) for i in t) + "\n")

    write_obj_with_quads(output_obj, verts, quads, triangles)
    return output_obj

def _sanitize_mesh(mesh: Trimesh) -> Trimesh:
    """Carefully sanitize a Trimesh to remove NaNs/Infs, degenerate faces,
    invalid UVs, and recompute normals so GLB/GLTF export doesn't produce
    JSON containing NaN values.
    """
    try:
        # Ensure arrays are numpy arrays
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64) if mesh.faces is not None else np.zeros((0, 3), dtype=np.int64)

        # Replace NaN/Inf in vertices
        if not np.all(np.isfinite(verts)):
            nlog('   - Replacing NaN/Inf in vertices with finite values')
            verts = np.nan_to_num(verts, nan=0.0, posinf=1e6, neginf=-1e6)

        # Remove faces that reference out-of-range indices after any changes
        max_vi = verts.shape[0] - 1
        valid_face_mask = np.all((faces >= 0) & (faces <= max_vi), axis=1) if faces.size else np.array([], dtype=bool)
        if faces.size and not np.all(valid_face_mask):
            nlog(f'   - Removing {np.count_nonzero(~valid_face_mask)} faces with invalid vertex indices')
            faces = faces[valid_face_mask]

        # Rebuild mesh with sanitized arrays
        mesh.vertices = verts
        if faces.size:
            mesh.faces = faces

        # Remove degenerate faces and unreferenced vertices (if trimesh provides them)
        fn = getattr(mesh, 'remove_degenerate_faces', None)
        if callable(fn):
            try:
                fn()
            except Exception as _e:
                nlog(f'   - remove_degenerate_faces failed during sanitize: {_e}')
        fn = getattr(mesh, 'remove_unreferenced_vertices', None)
        if callable(fn):
            try:
                fn()
            except Exception as _e:
                nlog(f'   - remove_unreferenced_vertices failed during sanitize: {_e}')

        # Ensure UVs are finite and in [0,1]
        uv = getattr(mesh.visual, 'uv', None)
        if uv is not None:
            try:
                uv = np.asarray(uv, dtype=np.float64)
                if not np.all(np.isfinite(uv)):
                    nlog('   - Replacing NaN/Inf in UVs with 0.0')
                    uv = np.nan_to_num(uv, nan=0.0, posinf=1.0, neginf=0.0)
                # clamp to [0,1]
                uv = np.clip(uv, 0.0, 1.0)
                # assign a proper TextureVisuals object instead of mutating ColorVisuals
                mesh.visual = TextureVisuals(uv=uv)
            except Exception as _e:
                nlog(f'   - UV sanitization failed: {_e}; dropping UVs')
                # Worst case: clear UVs by assigning an empty TextureVisuals
                mesh.visual = TextureVisuals()

        # Recompute/fix normals
        try:
            mesh.fix_normals()
        except Exception as _e:
            nlog(f'   - fix_normals failed during sanitize: {_e}')
            try:
                # fallback: set reasonable vertex normals from face normals
                _ = mesh.vertex_normals
            except Exception:
                pass

        return mesh
    except Exception as e:
        nlog(f'   - Mesh sanitization encountered an error: {e}')
        return mesh


def generate_uvs(mesh_path) -> Tuple[str, Trimesh]:
    """Stable UV generation: use trimesh.load(process=True) and avoid destructive API calls.

    This variant prefers XAtlas when available, falls back to a conservative UV placeholder,
    sanitizes the mesh (NaN/Inf clamp, UV clamp, normals recompute) and exports both OBJ
    and GLB. It intentionally avoids calling `remove_degenerate_faces` (or similar)
    operations that may remove valid faces on some trimesh installations.
    """
    nlog(">>> Stage 4: Professional UV Unwrapping & Smoothing...")
    nlog.progress('uv', percent=0, details={'phase': 'start', 'mesh': os.path.basename(mesh_path), 'gpu': _gpu_stats()})

    # Load mesh with processing enabled so trimesh can fix common topology issues
    mesh: Trimesh = cast(Trimesh, trimesh.load(mesh_path, process=True))

    # Defensive: trimesh.load may return a Scene; if so, pick the first geometry
    # so downstream operations (smoothing / xatlas) expect a Trimesh. Use getattr/cast(Any)
    # to avoid static analyzers complaining about 'geometry' attribute on Trimesh.
    try:
        if not hasattr(mesh, 'vertices') and hasattr(mesh, 'geometry'):
            try:
                scene_any = cast(Any, mesh)
                geometry = getattr(scene_any, 'geometry', None)
                if geometry:
                    first_geom = next(iter(geometry.values()))
                    mesh = cast(Trimesh, first_geom)
                else:
                    # fallback to an empty Trimesh to avoid attribute errors
                    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
            except Exception:
                # fallback to an empty Trimesh to avoid attribute errors
                mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    except Exception:
        # if the check itself fails, continue with what we have and guard later
        pass

    # 1) Smoothing: apply Taubin (guard for installs without smoothing)
    try:
        if getattr(trimesh, 'smoothing', None) is not None:
            trimesh.smoothing.filter_taubin(mesh, iterations=10)
            nlog('   - Applied Taubin smoothing (10 iterations)')
    except Exception as _e:
        nlog(f'   - Smoothing failed: {_e}')

    # 2) UV generation: prefer xatlas.parametrize when available
    final_mesh: Trimesh
    try:
        import xatlas
        try:
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            vmapping = np.asarray(vmapping)
            indices = np.asarray(indices)
            uvs = np.asarray(uvs)

            # vmapping may be indices into original vertices or explicit coordinates
            if vmapping.ndim == 1 or vmapping.dtype.kind in ('i', 'u'):
                try:
                    vmapped_verts = mesh.vertices[vmapping.astype(int)]
                except Exception:
                    vmapped_verts = mesh.vertices.copy()
            elif vmapping.ndim == 2 and vmapping.shape[1] == 3:
                vmapped_verts = vmapping
            else:
                vmapped_verts = mesh.vertices.copy()

            # indices may be flat or Nx3
            if indices.ndim == 1:
                try:
                    tri_faces = indices.reshape(-1, 3).astype(int)
                except Exception:
                    tri_faces = mesh.faces.copy()
            elif indices.ndim == 2 and indices.shape[1] == 3:
                tri_faces = indices.astype(int)
            else:
                tri_faces = mesh.faces.copy()

            final_mesh = trimesh.Trimesh(vertices=vmapped_verts, faces=tri_faces, process=True)

            # Attach UVs if they align to vertex count, otherwise create a conservative placeholder
            if uvs.ndim == 2 and len(uvs) == len(final_mesh.vertices):
                final_mesh.visual = TextureVisuals(uv=np.asarray(uvs))
            else:
                final_mesh.visual = TextureVisuals(uv=np.asarray(final_mesh.vertices[:, :2]))

            nlog('   - XAtlas parameterization applied')
        except Exception as _e:
            nlog(f'   - XAtlas param failed: {_e}; falling back to original mesh UVs')
            final_mesh = mesh.copy()
            if not getattr(final_mesh.visual, 'uv', None):
                final_mesh.visual = TextureVisuals(uv=np.asarray(final_mesh.vertices[:, :2]))
    except Exception as _e:
        nlog(f'XAtlas not available ({_e}); using original mesh UVs')
        final_mesh = mesh.copy()
        if not getattr(final_mesh.visual, 'uv', None):
            final_mesh.visual = TextureVisuals(uv=np.asarray(final_mesh.vertices[:, :2]))

    final_mesh = cast(Trimesh, final_mesh)

    # Sanitize using our helper which is robust and non-destructive
    nlog('   - Sanitizing mesh data (fixing NaNs / UVs / normals)...')
    nlog.progress('uv', percent=50, details={'phase': 'sanitizing', 'gpu': _gpu_stats()})
    try:
        final_mesh = _sanitize_mesh(final_mesh)
    except Exception as _e:
        nlog(f'   - _sanitize_mesh failed: {_e}')

    # Final guards: ensure finite arrays and clean UVs before export
    final_obj_path = mesh_path.replace('.obj', '_uv.obj')
    final_glb_path = mesh_path.replace('.obj', '_uv.glb')
    try:
        if not np.all(np.isfinite(final_mesh.vertices)):
            nlog('   - Final guard: replacing NaN/Inf in vertices before OBJ export')
            final_mesh.vertices = np.nan_to_num(final_mesh.vertices, nan=0.0, posinf=1e6, neginf=-1e6)

        vis = getattr(final_mesh, 'visual', None)
        uv = None
        if vis is not None:
            uv = getattr(vis, 'uv', None)
        if uv is not None:
            uv = np.asarray(uv)
            if not np.all(np.isfinite(uv)):
                nlog('   - Final guard: cleaning UVs before export')
                clean_uv = np.nan_to_num(uv, nan=0.0, posinf=1.0, neginf=0.0)
                final_mesh.visual = TextureVisuals(uv=clean_uv)

        cast(Trimesh, final_mesh).export(final_obj_path, include_normals=True)
        nlog(f'   - Exported OBJ: {final_obj_path}')
    except Exception as _e:
        nlog(f'   - Failed to export OBJ: {_e}')

    try:
        cast(Trimesh, final_mesh).export(final_glb_path)
        nlog(f'   - Exported GLB: {final_glb_path}')
    except Exception as _e:
        nlog(f'   ! GLB Export failed: {_e} (But OBJ is saved)')

    nlog.progress('uv', percent=100, details={'phase': 'done', 'mesh': os.path.basename(final_obj_path), 'gpu': _gpu_stats()})
    return final_obj_path, final_mesh

def bake_texture(raw_mesh_path, clean_mesh_path, resolution=2048):
    nlog(">>> Stage 5: Texture Baking (Projecting details)...")
    nlog.progress('bake', percent=0, details={'phase': 'start', 'gpu': _gpu_stats()})
    nlog.progress('bake', percent=0, details={'phase': 'start', 'gpu': _gpu_stats()})
    # Attempt to use Blender in headless mode to bake diffuse/albedo texture.
    # If Blender is not available, leave as a placeholder and return the mesh path.
    # Respect user-provided BLENDER_EXE env var first (allows nonstandard installs)
    blender_exe = os.environ.get('BLENDER_EXE') or None
    try:
        import shutil
        if not blender_exe:
            blender_exe = shutil.which('blender')
        if not blender_exe:
            # common Windows install path
            possible = [
                r"C:\Program Files\Blender Foundation\Blender\blender.exe",
                r"C:\Program Files (x86)\Blender Foundation\Blender\blender.exe",
            ]
            for p in possible:
                if os.path.exists(p):
                    blender_exe = p
                    break
    except Exception:
        # keep any env-provided value if present
        blender_exe = blender_exe or None

    out_tex = os.path.splitext(clean_mesh_path)[0] + "_albedo.png"
    if blender_exe:
        print(f"   Found Blender executable: {blender_exe} — running headless bake...")
        # write a small Blender Python script to perform a simple bake
        bake_script = f"""
import bpy, sys, os
argv = sys.argv
if '--' in argv:
    argv = argv[argv.index('--')+1:]
else:
    argv = []
raw = argv[0] if len(argv) > 0 else ''
clean = argv[1] if len(argv) > 1 else ''
out = argv[2] if len(argv) > 2 else 'baked.png'
res = int(argv[3]) if len(argv) > 3 else 2048

# clear
bpy.ops.wm.read_factory_settings(use_empty=True)

# import meshes
if raw and os.path.exists(raw):
    bpy.ops.import_scene.obj(filepath=raw)
raw_objs = [o for o in bpy.data.objects if o.type == 'MESH']

if clean and os.path.exists(clean):
    bpy.ops.import_scene.obj(filepath=clean)
clean_objs = [o for o in bpy.data.objects if o.type == 'MESH' and o not in raw_objs]

if not clean_objs:
    print('No clean mesh found for baking')
    sys.exit(1)

clean_obj = clean_objs[0]

# ensure UVs exist
bpy.context.view_layer.objects.active = clean_obj
bpy.ops.object.mode_set(mode='OBJECT')
if not clean_obj.data.uv_layers:
    clean_obj.data.uv_layers.new(name='UVMap')

# create image
img = bpy.data.images.new('BakeImage', width=res, height=res)
img.filepath_raw = out
img.file_format = 'PNG'

# create material and image texture node
mat = bpy.data.materials.new(name='BakeMat')
mat.use_nodes = True
nodes = mat.node_tree.nodes
tex_node = nodes.new('ShaderNodeTexImage')
tex_node.image = img
nodes.active = tex_node
if not clean_obj.data.materials:
    clean_obj.data.materials.append(mat)
else:
    clean_obj.data.materials[0] = mat

# select objects for baking: set clean as active and selected
bpy.ops.object.select_all(action='DESELECT')
clean_obj.select_set(True)
bpy.context.view_layer.objects.active = clean_obj

# set cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'

# perform bake (diffuse color)
bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

# save image
img.filepath_raw = out
img.save()
print('Baking complete ->', out)
"""

        import tempfile
        script_fd, script_path = tempfile.mkstemp(suffix="_bake.py")
        with os.fdopen(script_fd, 'w', encoding='utf8') as fh:
            fh.write(bake_script)

        try:
            subprocess.run([blender_exe, '--background', '--python', script_path, '--', raw_mesh_path, clean_mesh_path, out_tex, str(resolution)], check=True)
            if os.path.exists(out_tex):
                print(f"   Baked texture written to {out_tex}")
                try:
                    os.remove(script_path)
                except Exception:
                    pass
                return out_tex
            else:
                print("   Blender ran but bake output not found; falling back.")
        except Exception as e:
            print(f"   Blender baking failed ({e}), falling back to placeholder.")
        try:
            os.remove(script_path)
        except Exception:
            pass

    print("   (Texture baking logic placeholder - Blender not found or failed)")
    return clean_mesh_path
