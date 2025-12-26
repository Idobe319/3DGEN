@echo off
REM Launch the GUI from a cmd-friendly script. Run this from the repo root.
REM It uses the venv Activate script for cmd and then launches the in-process launcher.
set ROOT=%~dp0
pushd %ROOT%\..
call ".\.venv\Scripts\Activate.bat"
python "scripts\launch_gui.py"
popd
