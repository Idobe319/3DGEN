# Quickstart — NeoForge Local

1) Activate venv:

```powershell
& .\.venv\Scripts\Activate.ps1
```

2) Install Python deps:

```powershell
pip install -r requirements.txt
pip install -r gui/requirements.txt
```

3) Generate sample images (optional):

```powershell
python scripts/generate_samples.py
```

4) Run quick CLI test on sample:

```powershell
python run_local.py --input samples/example.jpg --poly 2500
```

5) To use TRELLIS weights:

```powershell
python scripts/download_trellis.py --url "<weights-url>" --dest tsr/model.ckpt
python run_local.py --input samples/example.jpg --trellis-weights tsr/model.ckpt
```

6) Run the desktop GUI:

```powershell
python gui/desktop_app.py
```

7) Packaging (Windows):

```powershell
# inside activated venv
pip install pyinstaller
python scripts/make_installer.bat
```

If you want, I can try to (a) fetch a weights URL you provide, (b) attempt to install Instant Meshes automatically (requires a valid `INSTANT_MESHES_URL`), or (c) test Blender headless baking (requires Blender installation).