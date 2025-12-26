import sys
import types
from PIL import Image
import run_local


def test_trellis_oom_retries_then_success(monkeypatch, tmp_path, capsys):
    # Create input
    inp = tmp_path / 'in.png'
    Image.new('RGBA', (32, 32), (10, 10, 10, 255)).save(inp)

    outdir = tmp_path
    mesh = tmp_path / 'temp_raw.obj'

    # Dummy GeometryGenerator to avoid fallback path being triggered
    class DummyGen:
        def __init__(self, *a, **k): pass
        def generate(self, processed):
            return []  # ensure TRELLIS path is used for mesh
    monkeypatch.setattr(run_local, 'GeometryGenerator', DummyGen)

    # Create fake Trellis pipeline that OOMs twice then succeeds
    class FakePipeline:
        counter = 0
        @classmethod
        def from_pretrained(cls, name):
            return cls()
        def cuda(self):
            return
        def cpu(self):
            return
        def run(self, image, **kwargs):
            FakePipeline.counter += 1
            if FakePipeline.counter < 3:
                raise RuntimeError('CUDA out of memory')
            class Mesh:
                def export(self, p):
                    open(p, 'w').write('#dummy')
            return {'mesh': [Mesh()]}

    # Install fake trellis modules into sys.modules so imports succeed
    fake_pipelines = types.ModuleType('trellis.pipelines')
    fake_pipelines.TrellisImageTo3DPipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'trellis', types.ModuleType('trellis'))
    monkeypatch.setitem(sys.modules, 'trellis.pipelines', fake_pipelines)

    # Run CLI with engine trellis
    monkeypatch.setattr(sys, 'argv', ['run_local.py', '--input', str(inp), '--engine', 'trellis', '--output', str(outdir)])
    run_local.main()
    captured = capsys.readouterr()
    assert 'TRELLIS OOM' in captured.out
    assert 'exported raw mesh' in captured.out or 'TRELLIS: exported raw mesh' in captured.out
    # ensure file was created
    assert mesh.exists()