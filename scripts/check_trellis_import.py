import sys, os, traceback
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print('python exe:', sys.executable)
print('sys.path[0]:', sys.path[0])
print('cwd:', os.getcwd())

# Check Open3D import
try:
    import open3d as o3d
    print('open3d ok, version:', getattr(o3d, '__version__', 'unknown'))
except Exception as e:
    print('open3d import FAILED')
    traceback.print_exc()

# Check TRELLIS import (from external/TRELLIS)
try:
    # Make sure external/TRELLIS is on path
    ext = os.path.join(project_root, 'external', 'TRELLIS')
    print('external/trellis path exists:', os.path.isdir(ext))
    sys.path.insert(0, ext)
    from trellis.pipelines import TrellisImageTo3DPipeline
    print('trellis import ok')
except Exception:
    print('trellis import FAILED')
    traceback.print_exc()

print('done')
