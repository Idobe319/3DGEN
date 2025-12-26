import os

from neoforge_core import preprocess_image, GeometryGenerator


def test_preprocess_and_generate(tmp_path):
    # Create a tiny image
    img = tmp_path / 'in.jpg'
    from PIL import Image
    Image.new('RGB', (64, 64), color=(255, 0, 0)).save(img)

    processed = preprocess_image(str(img))
    assert os.path.exists(processed)

    gen = GeometryGenerator()
    raw = gen.generate(processed)
    assert os.path.exists(raw)
