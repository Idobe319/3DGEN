"""Attempt to run Blender headless bake on sample mesh to validate setup.

This script will try to find Blender and run the bake script used by
`neoforge_core.bake_texture` and report success or failure.
"""
import os
from pathlib import Path

from neoforge_core import bake_texture


def find_blender():
    import shutil
    blender = shutil.which('blender')
    if blender:
        return blender
    common = [
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender\blender.exe",
    ]
    for p in common:
        if os.path.exists(p):
            return p
    return None


def main():
    # create tiny mesh
    tmp = Path('workspace')
    tmp.mkdir(exist_ok=True)
    mesh = tmp / 'test_mesh.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    blender = find_blender()
    print('Found blender:', blender)
    out = bake_texture(str(mesh), str(mesh), resolution=64)
    print('Bake result:', out)


if __name__ == '__main__':
    main()
