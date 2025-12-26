import torch
print('torch', getattr(torch, '__version__', None))
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print('device_count', torch.cuda.device_count())
        print('device_name', torch.cuda.get_device_name(0))
    except Exception as e:
        print('cuda_info_error', e)
else:
    try:
        import subprocess
        subprocess.run(['nvidia-smi'])
    except Exception as e:
        print('nvidia_smi_error', e)
