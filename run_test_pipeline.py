from neoforge_core import preprocess_image, GeometryGenerator, process_retopology, generate_uvs
import os

os.makedirs('workspace', exist_ok=True)
# pick a bundled dataset image
input_img = 'datasets/bunny_botsch.png'
if not os.path.exists(input_img):
    raise SystemExit('Dataset image not found: ' + input_img)

print('Using input:', input_img)
clean = preprocess_image(input_img)
print('Preprocessed ->', clean)

gen = GeometryGenerator()
raw = gen.generate(clean)
print('Generated raw mesh ->', raw)

clean_obj = os.path.join('workspace','clean_quads.obj')
process_retopology(raw, clean_obj, vertex_count=2500)
print('Retopology ->', clean_obj)

uv_path, mesh = generate_uvs(clean_obj)
print('UV export ->', uv_path)
print('Done')
