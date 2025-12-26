import sys

print('python:', sys.executable)

try:
    import torch
    print('torch.__version__:', getattr(torch, '__version__', 'unknown'))
    tv = getattr(torch, 'version', None)
    print('torch.version.cuda:', getattr(tv, 'cuda', 'unknown'))
    try:
        print('cuda_available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                print('cuda_device:', torch.cuda.get_device_name(0))
            except Exception as _e:
                print('cuda_device_error:', _e)
    except Exception as _e:
        print('cuda_query_error:', _e)
except Exception as e:
    print('torch_import_error:', e)

try:
    import flash_attn
    print('flash_attn.__file__:', getattr(flash_attn, '__file__', 'unknown'))
    print('has flash_attn_func:', hasattr(flash_attn, 'flash_attn_func'))
    try:
        import inspect
        print('flash_attn dir sample:', list(sorted(dir(flash_attn)))[:20])
    except Exception:
        pass
except Exception as e:
    print('flash_attn_import_error:', e)

try:
    import pkg_resources
    dist = None
    try:
        dist = pkg_resources.get_distribution('flash-attn')
    except Exception:
        pass
    print('pkg flash-attn dist:', getattr(dist, 'version', None))
except Exception:
    try:
        import pip
        print('pip available')
    except Exception:
        pass