import tempfile
import os
from PIL import Image
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from external.TripoSR.tsr.utils import save_video

frames = []
for i in range(10):
    arr = np.zeros((64,64,3), dtype=np.uint8)
    arr[...,0] = (i*25) % 255
    img = Image.fromarray(arr)
    frames.append(img)

fd, path = tempfile.mkstemp(suffix='.mp4')
import os
os.close(fd)
try:
    save_video(frames, path, fps=10)
    print('Wrote test video to', path, 'size', os.path.getsize(path))
except Exception as e:
    print('save_video failed:', e)
    # inspect created png frames
    base = path.rsplit('.',1)[0]
    pngs = [p for p in os.listdir(os.path.dirname(path)) if p.startswith(os.path.basename(base)) and p.endswith('.png')]
    print('pngs:', pngs[:5])
