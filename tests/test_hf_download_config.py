import subprocess
import sys
from pathlib import Path

def test_hf_download_config(tmp_path):
    dest = tmp_path / "config.yaml"
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'hf_download_trellis.py'
    cmd = [sys.executable, str(script), '--repo', 'stabilityai/TripoSR', '--config-dest', str(dest)]
    res = subprocess.run(cmd, check=False)
    assert res.returncode == 0, f"hf_download_trellis failed with code {res.returncode}"
    assert dest.exists() and dest.stat().st_size > 0
