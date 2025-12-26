"""Launch the desktop GUI in-process (avoids PowerShell path parsing issues).

Usage:
    python scripts/launch_gui.py
"""

try:
    from gui import desktop_app
except Exception as e:
    print('Failed to import gui.desktop_app:', e)
    raise

if __name__ == '__main__':
    desktop_app.main()
