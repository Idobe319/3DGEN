#!/usr/bin/env python3
"""Simple CLI to display TRELLIS expected checkpoint filenames and check a model dir."""
import argparse
import os
from scripts.trellis_utils import discover_expected_ckpts, check_model_dir_for_ckpts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', help='Path to local model directory containing ckpts/ (optional)')
    p.add_argument('--download-missing', action='store_true', help='Attempt authenticated download of missing ckpt files from Hugging Face')
    p.add_argument('--hf-token', help='Hugging Face token to use for authenticated downloads (optional)')
    args = p.parse_args()

    pr = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    expected = discover_expected_ckpts(pr)
    # allow hf_token via env if not passed
    if not hasattr(args, 'hf_token') or args.hf_token is None:
        args.hf_token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if not expected:
        print('No expected checkpoint filenames discovered in TRELLIS configs.')
        return
    print('Expected checkpoint filenames (from TRELLIS configs):')
    for fn in sorted(expected):
        print('  -', fn)

    if args.model_dir:
        md = os.path.abspath(args.model_dir)
        missing = check_model_dir_for_ckpts(md, expected)
        if missing:
            print('\nModel dir:', md)
            print('Missing ckpt files:')
            for fn in missing:
                print('  -', fn)
            # print HF hints for each missing file
            from scripts.trellis_utils import discover_ckpt_sources, ckpt_hint
            sources = discover_ckpt_sources(pr)
            print('\nHints for missing files (repo/path / HF URL):')
            for fn in missing:
                print('  -', ckpt_hint(fn, sources))

            if args.download_missing:
                print('\nAttempting to download missing files from Hugging Face...')
                from scripts.trellis_utils import download_missing_ckpts
                res = download_missing_ckpts(md, expected=expected, hf_token=args.hf_token)
                if 'error' in res:
                    print('Download failed to start:', res['error'])
                else:
                    if res['downloaded']:
                        print('Downloaded:')
                        for fn in res['downloaded']:
                            print('  -', fn)
                    if res['failed']:
                        print('Failed to download:')
                        for fn in res['failed']:
                            print('  -', fn)
                    # re-run missing check
                    missing2 = check_model_dir_for_ckpts(md, expected)
                    if missing2:
                        print('\nAfter download attempt, still missing:')
                        for fn in missing2:
                            print('  -', fn)
                    else:
                        print('\nAll expected ckpt files are now present.')
        else:
            print('\nModel dir:', md)
            print('All expected ckpt files are present.')

if __name__ == '__main__':
    main()
