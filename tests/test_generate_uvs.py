import os
import struct
import trimesh
from neoforge_core import generate_uvs


def check_glb_json_no_nan(glb_path: str) -> bool:
    with open(glb_path, 'rb') as f:
        b = f.read()
    if len(b) < 20:
        return False
    chunk_len = struct.unpack_from('<I', b, 12)[0]
    json_bytes = b[20:20+chunk_len]
    s = json_bytes.decode('utf-8', errors='replace')
    return ('NaN' not in s) and ('nan' not in s)


def test_generate_uvs_smoke(tmp_path):
    # Create a simple mesh (icosphere) and export to OBJ
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    obj_path = str(tmp_path / 'test_sphere.obj')
    mesh.export(obj_path)

    # Run generate_uvs
    uv_obj_path, out_mesh = generate_uvs(obj_path)

    # Basic sanity checks
    assert os.path.exists(uv_obj_path), 'UV OBJ not created'
    glb_path = uv_obj_path.replace('.obj', '.glb')
    assert os.path.exists(glb_path), 'UV GLB not created'
    assert hasattr(out_mesh, 'faces') and len(out_mesh.faces) > 0, 'Output mesh has no faces'

    # Ensure GLB JSON chunk contains no textual NaN tokens
    assert check_glb_json_no_nan(glb_path), 'GLB JSON contains textual NaN or nan'
