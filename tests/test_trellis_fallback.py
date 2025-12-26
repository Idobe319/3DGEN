import sys
import os
import subprocess
from PIL import Image

import run_local


def test_trellis_fallback_to_tripo(monkeypatch, tmp_path, capsys):
    # Create minimal input image
    inp = tmp_path / 'in.png'
    Image.new('RGBA', (32, 32), (255, 0, 0, 255)).save(inp)

    # Create a dummy obj that the mocked GeometryGenerator will return
    mesh = tmp_path / 'temp_raw.obj'
    mesh.write_text('# dummy obj')

    class DummyGen:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, processed):
            return [{'mesh_path': str(mesh), 'output_dir': str(tmp_path)}]

    # Monkeypatch the GeometryGenerator used by run_local to avoid heavy work
    monkeypatch.setattr(run_local, 'GeometryGenerator', DummyGen)

    # Simulate TRELLIS missing by injecting None into sys.modules so import fails gracefully
    monkeypatch.setitem(sys.modules, 'trellis', None)

    # Run main with --engine trellis to force TRELLIS path which should fall back
    monkeypatch.setattr(sys, 'argv', ['run_local.py', '--input', str(inp), '--engine', 'trellis', '--output', str(tmp_path)])

    # Execute
    run_local.main()

    captured = capsys.readouterr()
    assert ('TRELLIS import failed' in captured.out) or ('TRELLIS failed' in captured.out) or ('falling back to TripoSR' in captured.out)
    # Ensure final outputs mention the OBJ path
    assert str(mesh) in captured.out or 'raw mesh' in captured.out or 'raw mesh ->' in captured.out
