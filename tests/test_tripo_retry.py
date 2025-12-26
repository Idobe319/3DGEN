import os
import subprocess
from pathlib import Path
import shutil
import tempfile
from tsr.system import TripoSRInterface


def test_triposr_retries_on_oom(monkeypatch, tmp_path):
    # Prepare a minimal TripoSR repo layout
    tripo_root = tmp_path / 'external' / 'TripoSR'
    out_dir = tripo_root / 'workspace' / 'triposr_run' / '0'
    nested_out = out_dir / '0'
    nested_out.mkdir(parents=True, exist_ok=True)
    # Pre-create a fake mesh that will be discovered after a successful run
    mesh_path = nested_out / 'mesh.obj'
    mesh_path.write_text('o FakeMesh\n')

    # Create a fake input image
    input_img = tmp_path / 'in.png'
    input_img.write_bytes(b'PNG')

    calls = {'count': 0}

    def fake_run(cmd, cwd, check, capture_output, text, env):
        calls['count'] += 1
        if calls['count'] == 1:
            # Simulate CUDA OOM on first attempt
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr='CUDA out of memory')
        # On subsequent attempts, simulate success
        return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    tsi = TripoSRInterface(model_ref='stabilityai/TripoSR', tripo_root=str(tripo_root))
    out = tsi([str(input_img)])

    assert isinstance(out, list) and len(out) == 1
    assert 'mesh_path' in out[0]
    # Ensure we did retry at least once
    assert calls['count'] >= 2
