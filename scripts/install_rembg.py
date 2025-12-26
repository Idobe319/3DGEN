"""Helper to install the optional `rembg` package used for high-quality
background removal. Run from the project root inside the project's venv:

    python scripts/install_rembg.py

If you prefer GPU-enabled install (when supported), run:

    python -m pip install rembg[gpu]
"""
import sys
import subprocess

def main():
    try:
        print('Installing rembg into current Python environment...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rembg'])
        print('rembg installed successfully.')
    except subprocess.CalledProcessError:
        print('Failed to install rembg with default options. You can try:')
        print('  pip install rembg[gpu]')

if __name__ == '__main__':
    main()
