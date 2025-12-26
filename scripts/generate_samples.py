"""Generate small sample images for tests and examples."""
from PIL import Image
import os

os.makedirs('samples', exist_ok=True)
img = Image.new('RGB', (128, 128), color=(200, 120, 80))
img.save('samples/example.jpg')
print('Wrote samples/example.jpg')
