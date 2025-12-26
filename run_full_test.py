from neoforge_core import preprocess_image, GeometryGenerator, process_retopology, generate_uvs, bake_texture
import os

os.makedirs('workspace', exist_ok=True)
input_img = 'datasets/bunny_botsch.png'
print('Using:', input_img)
clean = preprocess_image(input_img)
print('preprocessed ->', clean)

g = GeometryGenerator()
raw = g.generate(clean)
print('raw ->', raw)

clean_obj = os.path.join('workspace','clean_quads.obj')
process_retopology(raw, clean_obj, vertex_count=2500, smooth_flow=True, im_iters=2, im_strength=0.5)
print('retopo ->', clean_obj)

uv, mesh = generate_uvs(clean_obj)
print('uv ->', uv)

baked = bake_texture(raw, uv, resolution=1024)
print('baked ->', baked)
