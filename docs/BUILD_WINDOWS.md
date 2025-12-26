# Windows Build & Troubleshooting Notes

This document collects common fixes for building native/CUDA components on Windows and tips for working with PowerShell.

## Visual Studio Developer Command Prompt
If you need to use the Visual Studio developer environment for native builds, ensure you invoke the DevCmd with proper arch flags:

```powershell
# Run in an elevated PowerShell if necessary
& "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
```

Note the `-arch=amd64` and `-host_arch=amd64` flags — these are required to select the correct toolset.

## PowerShell: running python commands
When invoking Python modules from PowerShell, always use the `&` operator with the executable path, e.g.:

```powershell
& "C:\Users\Admin\Desktop\3DGEN\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& "C:\Users\Admin\Desktop\3DGEN\.venv\Scripts\python.exe" -m pip show torch
```

Avoid putting the path inside quotes without the `&` prefix — PowerShell treats that as a string.

## Missing typing_extensions in base conda
If `python -c "import torch"` fails due to `ModuleNotFoundError: No module named 'typing_extensions'`, fix it by installing into the conda base env:

```powershell
conda install -n base typing_extensions -y
```

## spconv and GPU builds
- Building `spconv` and other CUDA-native libs on Windows is often fragile. If your tests are failing due to `CreateProcess failed` / `ninja fatal` during spconv build, prefer using WSL2.
- If you must build on Windows, ensure:
  - Visual Studio Build Tools + CMake installed
  - Ninja on PATH
  - CUDA toolkit matching your driver

## Recommendation
For TRELLIS/TripoSR development and to avoid Windows build issues, use WSL2 Ubuntu with GPU passthrough. See `docs/WSL_SETUP.md` for detailed instructions.
