"""
PyInstaller hook for MediaPipe to include model files and resources.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Collect all data files from MediaPipe
datas = collect_data_files('mediapipe', excludes=['**/__pycache__', '**/*.pyc'])

# Also collect submodules to ensure everything is included
hiddenimports = collect_submodules('mediapipe')

