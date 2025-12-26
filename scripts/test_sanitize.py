import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import gui.desktop_app as d
print(repr(d._sanitize_log('\x1b[31mred\x1b[0m')))
print(repr(d._sanitize_log('SOURCE LOADED: Screenshot 2025-02-20 202541.png\x1b')))
print('done')
