import sys
from run_local import main

if __name__ == '__main__':
    sys.argv = ['run_local.py', '--input', 'samples/example.jpg', '--poly', '2500']
    main()
