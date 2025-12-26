# NeoForge Desktop

Simple PySide6 desktop launcher for the NeoForge pipeline.

Usage

1. Activate your project's venv (example on Windows PowerShell):

```powershell
& C:\Users\Admin\Desktop\3DGEN\.venv\Scripts\Activate.ps1
```

2. Install GUI deps (if not already installed):

```powershell
pip install -r gui/requirements.txt
```

3. Run the app:

```powershell
python gui/desktop_app.py
```

Alternative (Windows): run the included launcher which activates the venv and runs the GUI:

```powershell
scripts\launch_gui.bat
```

The app calls `run_local.py` located at the project root and displays CLI output in a log window. Use the "Open output folder" button to inspect generated files in `workspace/`.
