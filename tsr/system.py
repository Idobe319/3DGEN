"""Stub implementation of TSR model for local testing.

Provides a `TSR.from_pretrained` that returns a DummyModel with the
methods used by `neoforge_core.py`: `.to()`, `__call__()` and
`.extract_mesh()`.

This is a development stub — replace with the real model and weights
for production use.
"""
from typing import Any, List
import trimesh
import os
import subprocess
from pathlib import Path


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device: str):
        # noop for stub
        return self

    def __call__(self, imgs: List[Any], device: str = "cpu"):
        # Return a trivial "scene code" object (we just pass through)
        return imgs

    def extract_mesh(self, scene_codes: List[Any]):
        # Produce a simple sphere mesh as a placeholder for each input
        meshes = []
        for _ in scene_codes:
            meshes.append(trimesh.primitives.Sphere(radius=1.0, subdivisions=4))
        return meshes


class TripoSRInterface:
    """Lightweight wrapper that invokes the external TripoSR `run.py` as a
    subprocess to perform inference and returns meshes back to the caller.

    This avoids importing the external `tsr` package into our process (name
    collisions and heavy C-extension requirements). It expects the TripoSR
    repository to be cloned at `external/TripoSR` relative to the project root
    (the `scripts/auto_setup.py` creates this), or otherwise available on the
    system PATH.
    """

    def __init__(self, model_ref: str, tripo_root: str = "external/TripoSR"):
        self.model_ref = model_ref
        # Resolve external TripoSR path relative to this project root
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / tripo_root
        if candidate.exists():
            self.tripo_root = candidate
        else:
            # fallback to raw tripo_root (might be absolute already)
            self.tripo_root = Path(tripo_root)

    def __call__(self, imgs: List[Any], device: str = "cpu", **kwargs) -> List[dict]:
        out_codes = []
        for i, im in enumerate(imgs):
            # Write the input image to a temporary folder for TripoSR to consume
            tmp_output = Path("workspace") / "triposr_run" / str(i)

            # Resolve output dir where external TripoSR will create files
            if tmp_output.is_absolute():
                tripo_output = tmp_output
            else:
                tripo_output = self.tripo_root / tmp_output

            tripo_output.mkdir(parents=True, exist_ok=True)
            input_path = tripo_output / "input.png"

            if isinstance(im, str):
                # already a path
                input_path.write_bytes(Path(im).read_bytes())
            else:
                # PIL Image-like or numpy array -> convert using PIL
                try:

                    im.save(input_path)
                except Exception:
                    # Fallback: if im is numpy array
                    import imageio

                    imageio.imwrite(str(input_path), im)

            # Build command to run external TripoSR. Use absolute output dir to avoid surprises.
            import sys
            cmd = [
                sys.executable,
                "run.py",
                str(input_path),
                "--pretrained-model-name-or-path",
                self.model_ref,
                "--output-dir",
                str(tripo_output),
                "--model-save-format",
                "obj",
            ]
            # Allow runtime options like mc_resolution, chunk_size, bake_texture, texture_resolution, render
            if 'mc_resolution' in kwargs and kwargs['mc_resolution']:
                cmd.extend(["--mc-resolution", str(kwargs['mc_resolution'])])
            else:
                # Use a higher default MC resolution for smoother geometry
                cmd.extend(["--mc-resolution", "1024"])
            if 'chunk_size' in kwargs and kwargs['chunk_size']:
                cmd.extend(["--chunk-size", str(kwargs['chunk_size'])])
            else:
                # Increase chunk size to avoid fractures when using higher resolution
                cmd.extend(["--chunk-size", "8192"])
            if kwargs.get('bake_texture') or kwargs.get('bake'):
                cmd.append("--bake-texture")
                tr = kwargs.get('texture_resolution') or kwargs.get('texture_resolution')
                if tr:
                    cmd.extend(["--texture-resolution", str(tr)])
            if kwargs.get('render'):
                cmd.append("--render")

            # Diagnostic: show command for debugging and capture output for robust errors
            try:
                print('TripoSR: running command:', ' '.join(cmd))
            except Exception:
                pass

            # Robust invocation with retries for CUDA OOMs and fallbacks
            def _run_subprocess_with_retries(cmd, cwd, attempts=4):
                """Run the TripoSR subprocess with sensible retries:
                1) original run
                2) set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and retry
                3) lower mc-resolution and retry
                4) force CPU fallback (CUDA_VISIBLE_DEVICES='')
                Returns the CompletedProcess on success or raises CalledProcessError on final failure.
                """
                base_env = os.environ.copy()

                def _exec(cmd_list, env):
                    res = subprocess.run(cmd_list, cwd=cwd, check=True, capture_output=True, text=True, env=env)
                    if res.stdout:
                        print('TripoSR stdout:', res.stdout)
                    if res.stderr:
                        print('TripoSR stderr:', res.stderr)
                    return res

                # 1: Try original
                try:
                    return _exec(cmd, base_env)
                except subprocess.CalledProcessError as e:
                    out = getattr(e, 'stdout', '') or ''
                    err = getattr(e, 'stderr', '') or ''
                    print(f"TripoSR failed (returncode={e.returncode}) on attempt 1. STDERR:\n{err}")

                    oom_keywords = ['CUDA out of memory', 'OutOfMemoryError', 'torch.cuda.OutOfMemoryError']
                    if not any(k in err or k in out for k in oom_keywords):
                        # Non-OOM failure — rethrow for existing handling
                        raise

                # 2: Retry with PYTORCH_CUDA_ALLOC_CONF
                try:
                    env2 = base_env.copy()
                    env2['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    print('Retrying TripoSR with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (attempt 2)')
                    return _exec(cmd, env2)
                except subprocess.CalledProcessError as e:
                    out = getattr(e, 'stdout', '') or ''
                    err = getattr(e, 'stderr', '') or ''
                    print(f"TripoSR retry with PYTORCH_CUDA_ALLOC_CONF failed (attempt 2). STDERR:\n{err}")
                    # If still not OOM, rethrow
                    if not any(k in err or k in out for k in oom_keywords):
                        raise

                # 3: Retry with reduced mc-resolution
                try:
                    cmd3 = list(cmd)
                    # helper to set or append an arg
                    def _set_arg(lst, key, val):
                        if key in lst:
                            idx = lst.index(key)
                            lst[idx+1] = str(val)
                        else:
                            lst.extend([key, str(val)])
                    # find current resolution and halve it
                    if '--mc-resolution' in cmd3:
                        idx = cmd3.index('--mc-resolution')
                        try:
                            cur = int(cmd3[idx+1])
                            new = max(256, int(cur // 2))
                            cmd3[idx+1] = str(new)
                        except Exception:
                            cmd3 = cmd3
                    else:
                        cmd3.extend(['--mc-resolution', '512'])
                    # also reduce chunk size if present
                    if '--chunk-size' in cmd3:
                        idx = cmd3.index('--chunk-size')
                        try:
                            curc = int(cmd3[idx+1])
                            newc = max(1024, int(curc // 4))
                            cmd3[idx+1] = str(newc)
                        except Exception:
                            pass
                    print('Retrying TripoSR with reduced mc-resolution/chunk-size (attempt 3):', ' '.join(cmd3))
                    env3 = base_env.copy()
                    env3['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    return _exec(cmd3, env3)
                except subprocess.CalledProcessError as e:
                    out = getattr(e, 'stdout', '') or ''
                    err = getattr(e, 'stderr', '') or ''
                    print(f"TripoSR retry with reduced resolution failed (attempt 3). STDERR:\n{err}")

                # 4: CPU fallback
                try:
                    env4 = base_env.copy()
                    env4['CUDA_VISIBLE_DEVICES'] = ''
                    env4['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    print('Retrying TripoSR on CPU (CUDA_VISIBLE_DEVICES="") (attempt 4) — this may be slow')
                    return _exec(cmd, env4)
                except subprocess.CalledProcessError as e:
                    out = getattr(e, 'stdout', '') or ''
                    err = getattr(e, 'stderr', '') or ''
                    print(f"TripoSR CPU fallback failed (attempt 4). STDERR:\n{err}")
                    # Give up — rethrow last exception
                    raise

            try:
                res = _run_subprocess_with_retries(cmd, cwd=str(self.tripo_root))
            except subprocess.CalledProcessError as e:
                out = getattr(e, 'stdout', '') or ''
                err = getattr(e, 'stderr', '') or ''
                diag = f"TripoSR failed (returncode={getattr(e,'returncode', 'unknown')})\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                print(diag)
                # If the run failed during rendering/encoding but meshes were produced, treat as warning and continue.
                search_root = tripo_output
                mesh_files = list(search_root.rglob("*.obj"))
                if not mesh_files:
                    # also check search_root/0/mesh.obj (older behavior)
                    alt = search_root / "0" / "mesh.obj"
                    if alt.exists():
                        mesh_files = [alt]
                if mesh_files:
                    print(f"Warning: TripoSR subprocess failed during post-processing, but found mesh outputs under {search_root}. Continuing with available mesh files.")
                else:
                    raise RuntimeError(diag) from e

            # After running, TripoSR places meshes under the given output dir (sometimes nested).
            # Search for any .obj file under that directory and pick the first one.
            search_root = tripo_output
            mesh_files = list(search_root.rglob("*.obj"))
            if not mesh_files:
                # also check search_root/0/mesh.obj (older behavior)
                alt = search_root / "0" / "mesh.obj"
                if alt.exists():
                    mesh_files = [alt]

            if not mesh_files:
                    # diagnostic listing (first 20 entries) to aid debugging
                    found = list(search_root.rglob("*"))
                    diag = ", ".join(str(p) for p in found[:20])
                    print("TripoSR output files:", diag)
                    raise RuntimeError(f"No mesh output found in {search_root}. See logs for file listing.")
            mesh_path = mesh_files[0]
            out_codes.append({"mesh_path": str(mesh_path), "output_dir": str(search_root)})
        return out_codes

    def extract_mesh(self, scene_codes: List[dict]):
        meshes = []
        for code in scene_codes:
            mp = Path(code["mesh_path"])
            if mp.exists():
                meshes.append(trimesh.load(mp, force="mesh"))
        return meshes


class TSR:
    @staticmethod
    def from_pretrained(*args, weight_path: str = None, url: str = None, **kwargs):
        """Attempt to use a real TripoSR model when possible.

        Behavior:
        - If `weight_path` points to a file, this will attempt to run the
          external TripoSR `run.py` using the HF repo `stabilityai/TripoSR` as
          the model reference (this will let TripoSR download any missing
          config files from HF while using the local cache for weights when
          available).
        - If no weights are found, fall back to DummyModel as before.
        """
        # Determine candidate weight locations
        candidates = []
        if weight_path:
            candidates.append(weight_path)
        candidates.append(os.path.join(os.path.dirname(__file__), "model.ckpt"))

        for p in candidates:
            try:
                if os.path.exists(p):
                    size = os.path.getsize(p)
                    print(f"TSR: found weights at {p} (size={size} bytes).")
                    print("Attempting to use external TripoSR for inference.")
                    # Use HF repo as the model reference; this lets the external
                    # TripoSR code fetch config.yaml if needed while reusing
                    # local cache for large weights.
                    model_ref = os.environ.get("TRELLIS_HF_REPO", "stabilityai/TripoSR")
                    return TripoSRInterface(model_ref)
            except Exception:
                pass

        # No weights found -> fallback to dummy
        print(
            "TSR: no weights found. Using DummyModel stub. To load real model provide --trellis-weights or run scripts/hf_download_trellis.py"
        )
        return DummyModel()
