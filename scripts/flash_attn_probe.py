import sys
import glob

print('CWD=', __import__('os').getcwd())

try:
    import flash_attn
    print('FILE=', getattr(flash_attn, '__file__', None))
    print('DIR_HAS_flash_attn_func=', 'flash_attn_func' in dir(flash_attn))
    print('DIR_SAMPLE=', [x for x in dir(flash_attn) if 'attn' in x or 'flash' in x][:40])
except Exception as e:
    print('flash_attn import error:', repr(e))

local_py = glob.glob('**/flash_attn.py', recursive=True)[:20]
local_pkg = glob.glob('**/flash_attn', recursive=True)[:20]
print('LOCAL flash_attn.py=', local_py)
print('LOCAL flash_attn folder=', local_pkg)
