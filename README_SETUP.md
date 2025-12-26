NeoForge 3D - Setup Notes

What I installed/added automatically:
- Created `neoforge_core.py` and `app.py` in the project root.
- Installed Python packages (see `requirements.txt`).
- Checked CUDA (nvcc 11.8 found).
- Cloned `instant-meshes` source and downloaded prebuilt Windows binaries and datasets.
- Extracted `Instant Meshes.exe` and `datasets` into the project root.

Files you must provide / manual steps:
1) TRELLIS / TripoSR model (required for `GeometryGenerator`):
   - Download the model code and weights (e.g., from StabilityAI / HuggingFace).
   - Place the module package in `tsr/` under the project root so that `from tsr.system import TSR` works.
   - Required files typically: `tsr/` package, `config.yaml`, and `model.ckpt`.
   - If the model is hosted on HuggingFace and requires authentication, use `huggingface-cli login` then `git lfs` to pull large files.

2) (Optional) If you prefer to build Instant Meshes yourself:
   - Install Visual Studio and CMake, then open `instant-meshes-src/InstantMeshes.sln` and build.

Quick run (after you provide TRELLIS files):
1. Activate your virtualenv (if not already):

   ```powershell
   & .\.venv\Scripts\Activate.ps1
   ```

2. Install requirements (if needed):

   ```powershell
   pip install -r requirements.txt
   ```

3. Run the app:

   ```powershell
   python app.py
   ```

New helper scripts and CLI options:
- To download TRELLIS weights directly:

```powershell
python scripts/download_trellis.py --url "<weights-url>" --dest tsr/model.ckpt
```

- `run_local.py` accepts these extra options:
  - `--trellis-weights` — path to local `model.ckpt` to use (if a single `.ckpt` file is provided, `run_local.py` will attempt to adapt it into a temporary model folder and load TRELLIS from that folder)
  - `--trellis-url` — URL to download weights before running

- To detect or install Instant Meshes (if you set `INSTANT_MESHES_URL` environment variable):

```powershell
python scripts/install_instant_meshes.py --download-if-missing
```

- For headless baking, install Blender and ensure `blender` is on PATH or set the full path.

GUI: use `scripts\launch_gui.bat` on Windows to activate the venv and open the desktop app.

To test Blender headless baking after installing Blender, run:

```powershell
python scripts/test_blender_bake.py
```

- The Gradio UI now disables the interactive `Model3D` preview by default; enable it in the UI if you want an in-browser preview (may show WebGPU warnings in some browsers).

Packaging (Windows):
- There is a placeholder `scripts/make_installer.bat` showing how to use `pyinstaller` to bundle `gui/desktop_app.py` into a single executable. This is a starting point — a real installer should include runtime dependency checks and shortcuts.


Notes & next steps:
- I did not download or configure the TripoSR/TRELLIS model automatically because weights are large and often require credentials or acceptance of model terms.
- If you give me a direct download URL for `model.ckpt` (or a HuggingFace repo URL and an access token), I can fetch and place it into `tsr/` for you.
- I can also attempt to build Instant Meshes from source if you want that instead of using the prebuilt binary.

TRELLIS import helper & editable install ✅
- If you want `import trellis` to work from any shell (including `python -c "import trellis"`), either set `PYTHONPATH` to the local external package or install TRELLIS as an editable package in your environment.

  Quick checks:

  - PowerShell (temporary PYTHONPATH):

    $env:PYTHONPATH = "$PWD\external\TRELLIS"
    conda run -n trellis python -c "import trellis; print(trellis.__file__)"

  - Install editable (recommended):

    conda run -n trellis python -m pip install -e .\external\TRELLIS
    conda run -n trellis python -c "import trellis; print(trellis.__file__)"

- I added helper scripts to make verification easier:
  - `scripts/install_trellis_editable.py` — installs `external/TRELLIS` in the active env as editable and verifies import.
  - `scripts/check_trellis_imports.py` — local diagnostic that suggests PYTHONPATH/editable installation and shows the exact command to run in your `trellis` env.
  - `scripts/run_trellis_verify.ps1` — PowerShell helper that sets the SDPA env vars, runs the canonical verification command in the `trellis` env, writes `workspace_trellis_real/run.log`, and prints the log context around key markers so you can paste it easily.

## Continuous Integration (GitHub Actions)
A minimal CI workflow has been added to run unit tests and probe for FlashAttention availability on pushes/PRs: `.github/workflows/ci.yml`.
The workflow installs dependencies, runs `pytest`, then runs `scripts/flash_attn_probe.py` and `scripts/detect_env.py` to record environment diagnostics.


## WSL2 setup for reliable FlashAttention
If you prefer to run TRELLIS under WSL2 for best FlashAttention support and fewer Windows build issues, see `docs/WSL_SETUP.md` for a step-by-step guide and use the scripts:

- `scripts/setup_wsl.ps1` — Windows-side helper (manual admin step required: run PowerShell as Administrator and run `wsl --install -d Ubuntu`)
- `scripts/wsl_setup.sh` — Run inside WSL Ubuntu to set up a Python environment (defaults to a `python3.10` venv, or pass `--use-conda` to use Miniconda). The script installs PyTorch (CU121), attempts `flash-attn` and `spconv-cu121` when appropriate, copies the project into the native WSL FS and runs environment probes.
- `scripts/install_flash_attn.sh` — helper that attempts pip install of `flash-attn` and falls back to building from source.
- `scripts/wsl_smoke_test.sh` — quick smoke tests (`nvidia-smi`, torch, flash_attn checks)- `scripts/wsl_install_helper.ps1` — Windows helper to invoke the WSL install (Admin required)
- `scripts/try_install_flash_attn_windows.ps1` — best-effort helper to attempt installing `flash-attn` into the Windows venv (may fail; WSL recommended)
- `scripts/smoke_pipeline_cpu.sh` — CPU-only smoke runner that executes a fast local pipeline run on a tiny synthetic image

Use `scripts/run_trellis_verify.ps1` to run the verification and it will print snippets around the lines we care about ("Loading TRELLIS pipeline" / "Running TRELLIS inference" / errors).
