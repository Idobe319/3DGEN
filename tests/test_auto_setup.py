import subprocess
import sys


def test_auto_setup_runs_ok():
    rc = subprocess.call([sys.executable, 'scripts/auto_setup.py'])
    assert rc == 0
