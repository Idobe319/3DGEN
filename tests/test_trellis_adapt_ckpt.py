import os
import tempfile
import shutil


def test_adapt_ckpt_to_tempdir(tmp_path):
    # create a fake checkpoint file
    ckpt = tmp_path / 'model.ckpt'
    ckpt.write_bytes(b'dummy')

    # simulate the adaptation logic exactly as in run_local
    tmpdir = tempfile.mkdtemp(prefix='trellis_ckpt_test_')
    try:
        ckpt_dir = os.path.join(tmpdir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        dest = os.path.join(ckpt_dir, os.path.basename(str(ckpt)))
        shutil.copyfile(str(ckpt), dest)
        # Ensure the adapted dir looks like a model folder
        assert os.path.isdir(tmpdir)
        assert os.path.isdir(ckpt_dir)
        assert os.path.isfile(dest)
        assert os.path.basename(tmpdir).startswith('trellis_ckpt_test_')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
