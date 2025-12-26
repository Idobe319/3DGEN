"""Helper to detect or download an Instant Meshes binary.

If INSTANT_MESHES_URL env var is set, it will attempt to download the archive
and extract the executable into the project root. This is platform-dependent
and currently implements a simple Windows extraction if a zip is provided.
"""
import os
import sys
import argparse

try:
    import requests
except Exception:
    print('Install script requires `requests` (pip install requests)')
    sys.exit(1)


def find_exe():
    possible = [
        os.path.join(os.getcwd(), "Instant Meshes.exe"),
        os.path.join(os.getcwd(), "InstantMeshes.exe"),
        os.path.join(os.getcwd(), "instant-meshes.exe"),
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    return None


def download_and_extract(url):
    dest = 'instant_meshes.zip'
    print(f'Downloading {url} -> {dest}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as fh:
            for chunk in r.iter_content(8192):
                fh.write(chunk)
    # Try to extract with zipfile
    try:
        import zipfile
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall('.')
        print('Extracted archive to project root.')
    except Exception as e:
        print('Failed to extract archive:', e)
    finally:
        try:
            os.remove(dest)
        except Exception:
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--download-if-missing', action='store_true')
    args = p.parse_args()

    found = find_exe()
    if found:
        print('Found Instant Meshes at', found)
        sys.exit(0)

    if args.download_if_missing and os.environ.get('INSTANT_MESHES_URL'):
        download_and_extract(os.environ['INSTANT_MESHES_URL'])
        found = find_exe()
        if found:
            print('Installed Instant Meshes at', found)
            sys.exit(0)
        else:
            print('Download completed but executable not found. Please inspect the archive contents.')
            sys.exit(2)

    print('Instant Meshes not found. Set INSTANT_MESHES_URL env var and run this script with --download-if-missing to attempt an install.')
    sys.exit(1)
