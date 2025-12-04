"""
PyInstaller hook for PyAutoGUI to include macOS dependencies.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect PyAutoGUI data files
try:
    datas = collect_data_files('pyautogui', excludes=['**/__pycache__', '**/*.pyc'])
except Exception:
    datas = []

# Collect all submodules (with error handling)
try:
    hiddenimports = collect_submodules('pyautogui')
except Exception:
    hiddenimports = ['pyautogui']  # Fallback to just the main module

# macOS-specific: Ensure Quartz framework is included
hiddenimports += [
    'Quartz',
    'rubicon.objc',
    'rubicon.objc.runtime',
    'rubicon.objc.types',
]


