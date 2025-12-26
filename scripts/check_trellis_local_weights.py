import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Ensure local external/TRELLIS is discoverable
sys.path.insert(0, os.path.join(project_root, 'external', 'TRELLIS'))
from trellis.pipelines import TrellisImageTo3DPipeline

# Simulate local path selection without running full pipeline
candidates = [
    os.path.join(project_root, 'tsr', 'model.ckpt'),
    os.path.join(project_root, 'tsr', 'trellis_weights'),
]
for p in candidates:
    if os.path.exists(p):
        print('Found candidate local TRELLIS path:', p)
        try:
            _ = TrellisImageTo3DPipeline.from_pretrained(p)
            print('Pipeline loaded from local path (OK):', p)
        except Exception as e:
            print('Pipeline failed to load from local path:', p)
            print('  error:', e)
            # If it's a single checkpoint file, try adapting it into a temporary model folder
            if os.path.isfile(p) and p.lower().endswith(('.ckpt', '.pt')):
                try:
                    import tempfile, shutil
                    tmpdir = tempfile.mkdtemp(prefix='trellis_ckpt_')
                    ckpt_dir = os.path.join(tmpdir, 'ckpts')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_name = os.path.basename(p)
                    dest = os.path.join(ckpt_dir, ckpt_name)
                    shutil.copyfile(p, dest)
                    print('Attempting to load pipeline from adapted temp model dir:', tmpdir)
                    _ = TrellisImageTo3DPipeline.from_pretrained(tmpdir)
                    print('Pipeline loaded from adapted temp model dir (OK):', tmpdir)
                except Exception as e2:
                    print('Adapted load also failed:', e2)
    else:
        print('Candidate not present:', p)

print('Done')
