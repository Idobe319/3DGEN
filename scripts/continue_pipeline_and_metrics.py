import os
import glob
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Ensure project root is on sys.path so imports work when run from scripts/
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from neoforge_core import process_retopology, generate_uvs, bake_texture, preprocess_image

mesh_in = 'external/TripoSR/workspace/triposr_run/0/0/mesh.obj'
out_retopo = 'workspace/clean_quads.obj'

os.makedirs('workspace', exist_ok=True)
print('Running retopology...')
try:
    process_retopology(mesh_in, out_retopo, vertex_count=2500)
    print('Retopology output:', out_retopo)
except Exception as e:
    print('Retopology failed:', e)

print('Generating UVs...')
try:
    uv_path, mesh = generate_uvs(out_retopo)
    print('UV output:', uv_path)
except Exception as e:
    print('UV generation failed:', e)
    uv_path = out_retopo

print('Running bake (may be placeholder if Blender missing)...')
try:
    baked = bake_texture(mesh_in, uv_path, resolution=2048)
    print('Bake output:', baked)
except Exception as e:
    print('Bake failed:', e)
    baked = uv_path

# Prepare reference image
print('Preprocessing original image...')
ref_path = preprocess_image('datasets/bunny_botsch.png')
ref_img = Image.open(ref_path).convert('RGB')
ref_arr = np.array(ref_img).astype(np.float32) / 255.0

# Determine resampling filter for Pillow versions (avoid direct attribute access to satisfy static analyzers)
_resampling = getattr(Image, 'Resampling', None)
if _resampling is not None:
    RESAMPLE_BILINEAR = getattr(_resampling, 'BILINEAR', getattr(Image, 'BILINEAR', 2))
else:
    RESAMPLE_BILINEAR = getattr(Image, 'BILINEAR', 2)

# Collect render frames
renders = sorted(glob.glob('external/TripoSR/workspace/triposr_run/0/0/render_*.png'))
if not renders:
    print('No render frames found; aborting metrics')
    raise SystemExit(1)

ssim_vals = []
for rpath in renders:
    rimg = Image.open(rpath).convert('RGB')
    # resize to match reference
    rimg = rimg.resize(ref_img.size, RESAMPLE_BILINEAR)
    rarr = np.array(rimg).astype(np.float32) / 255.0
    # compute grayscale SSIM
    from skimage.color import rgb2gray
    s = ssim(rgb2gray(ref_arr), rgb2gray(rarr), data_range=1.0)
    ssim_vals.append(s)

avg_ssim = sum(ssim_vals) / len(ssim_vals)
print(f'Average SSIM over {len(ssim_vals)} renders: {avg_ssim:.4f}')

# LPIPS (optional)
try:
    import importlib
    lpips_mod = importlib.import_module('lpips')
    import torch
    loss_fn = lpips_mod.LPIPS(net='alex')
    lpips_vals = []
    for rpath in renders:
        rimg = Image.open(rpath).convert('RGB')
        rimg = rimg.resize(ref_img.size, RESAMPLE_BILINEAR)
        a = np.array(ref_img).astype(np.float32) / 255.0
        b = np.array(rimg).astype(np.float32) / 255.0
        # convert to torch tensors [1,3,H,W], range [-1,1]
        ta = torch.from_numpy(a).permute(2,0,1).unsqueeze(0) * 2 - 1
        tb = torch.from_numpy(b).permute(2,0,1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            d = loss_fn(ta, tb).item()
        lpips_vals.append(d)
    avg_lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else None
    print(f'Average LPIPS over {len(lpips_vals)} renders: {avg_lpips:.4f}')
except Exception as e:
    print('LPIPS not available or failed:', e)

print('Done. Outputs:')
print('  retopo:', out_retopo)
print('  uv:', uv_path)
print('  baked:', baked)
print('  renders (count):', len(renders))
