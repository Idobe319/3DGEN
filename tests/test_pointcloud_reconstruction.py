import os
import tempfile
import numpy as np
import trimesh
import pytest
pytest.importorskip("open3d")
from neoforge_core import process_retopology


def test_pointcloud_reconstruction_creates_mesh(tmp_path):
    # create a point-cloud-only OBJ (v lines only)
    pts = np.random.randn(2000, 3) * 0.5
    in_obj = tmp_path / "points.obj"
    out_obj = tmp_path / "reconstructed.obj"
    with open(in_obj, 'w', encoding='utf8') as f:
        for p in pts:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    res = process_retopology(str(in_obj), str(out_obj), vertex_count=1500)
    assert os.path.exists(res)
    mesh = trimesh.load(res, process=False)
    # We expect a reconstructed mesh to have faces
    assert hasattr(mesh, 'faces') and len(mesh.faces) > 0, "Reconstruction did not produce faces"
