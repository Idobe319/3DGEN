from tsr.system import TSR


def test_from_pretrained_finds_local_weights(tmp_path, capsys):
    # create a fake weights file
    models_dir = tmp_path / "tsr"
    models_dir.mkdir()
    w = models_dir / "model.ckpt"
    w.write_bytes(b"fake-weights")

    # call from_pretrained with explicit path
    m = TSR.from_pretrained('stabilityai/TripoSR', weight_path=str(w))
    assert m is not None
    captured = capsys.readouterr()
    assert 'found weights' in captured.out or 'no weights found' not in captured.out
