import sys, os
print('executable:', sys.executable)
print('cwd:', os.getcwd())
print('sys.path[0]:', sys.path[0])
print('gui exists:', os.path.exists('gui'))
print('top-level:', os.listdir('.'))
