from neoforge_core import bake_texture


def test_bake_fallback(tmp_path):
    # create a small dummy mesh file (simple OBJ)
    mesh = tmp_path / 'mesh.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    out = bake_texture(str(mesh), str(mesh), resolution=64)
    # since Blender is not expected in CI, the function should return the mesh path
    assert out == str(mesh) or out.endswith('_albedo.png')
