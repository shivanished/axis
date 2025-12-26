# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_all
opencv_binaries, opencv_datas, opencv_hiddenimports = collect_all("cv2")
opencv_binaries = [(b, ".") for b, _ in opencv_binaries]


# ----------------------------------------------------------------------
# Hard-disable Qt WebEngine (not used)
# ----------------------------------------------------------------------
sys.modules["PySide6.QtWebEngineCore"] = None
sys.modules["PySide6.QtWebEngineWidgets"] = None
sys.modules["PySide6.QtWebEngineQuick"] = None

block_cipher = None
spec_root = os.path.dirname(os.path.abspath(SPEC))

# ----------------------------------------------------------------------
# MediaPipe model data ONLY (NO mediapipe import)
# ----------------------------------------------------------------------
mediapipe_datas = []
try:
    import pkgutil

    spec = pkgutil.get_loader("mediapipe")
    if spec and spec.path:
        mp_root = os.path.dirname(spec.path)
        for root, dirs, files in os.walk(mp_root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".tflite", ".pbtxt", ".pb", ".bin", ".binarypb", ".txt")):
                    rel = os.path.relpath(root, mp_root)
                    mediapipe_datas.append(
                        (os.path.join(root, f), os.path.join("mediapipe", rel))
                    )
except Exception as e:
    print(f"Warning: MediaPipe models not collected: {e}")

# ----------------------------------------------------------------------
# Datas
# ----------------------------------------------------------------------
datas = []
datas.extend(mediapipe_datas)

# ----------------------------------------------------------------------
# Hidden imports (lean + correct)
# ----------------------------------------------------------------------
hiddenimports = [
    # App
    "main",
    "gesture_controller",
    "settings",
    "gui",
    "gui.main_window",
    "gui.overlay_window",
    "gui.settings_dialog",
    "gui.camera_widget",
    "gui.control_panel",

    # OpenCV / NumPy (ORDER MATTERS)
    "cv2",
    "numpy",

    # MediaPipe
    "mediapipe",
    "mediapipe.python.solutions.hands",
    "mediapipe.python.solutions.drawing_utils",

    # Qt
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",

    # Input / OS
    "pyautogui",
    "pyautogui._pyautogui_osx",
    "Quartz",
    "AppKit",
    "Foundation",

    # Utilities
    "pynput",
    "pynput.mouse",
    "pynput.keyboard",
    "PIL.Image",
]

# Let PyInstaller expand MediaPipe internals safely
try:
    hiddenimports.extend(collect_submodules("mediapipe"))
except Exception:
    pass

# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
a = Analysis(
    ["main.py"],
    pathex=[spec_root],
    binaries=opencv_binaries,
    datas=datas + opencv_datas,
    hiddenimports=hiddenimports + opencv_hiddenimports,
    hookspath=[os.path.join(spec_root, "hooks")],
    runtime_hooks=[
        os.path.join(spec_root, "hooks", "pyautogui_runtime.py")
    ],
    excludes=[
        # Qt junk
        "PyQt5",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",

        # Qt3D (framework symlink crash source)
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DAnimation",
        "Qt3DCore",
        "Qt3DRender",
        "Qt3DInput",
        "Qt3DLogic",
        "Qt3DAnimation",

        # Tests
        "tkinter",
        "nltk",
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
    ],
    cipher=block_cipher,
    noarchive=False,
)

# ----------------------------------------------------------------------
# Build stages
# ----------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Axis",
    console=False,
    strip=False,
    upx=True,
    entitlements_file=os.path.join(
        spec_root, "gesture_control.entitlements"
    ),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Axis_collect",
)

app = BUNDLE(
    coll,
    name="Axis.app",
    icon=os.path.join(spec_root, "app_icon.icns")
    if os.path.exists(os.path.join(spec_root, "app_icon.icns"))
    else None,
    bundle_identifier="com.axis.app",
    version="1.0.0",
    info_plist={
        "NSCameraUsageDescription": "Axis needs camera access to track hand gestures.",
        "NSAccessibilityUsageDescription": "Axis needs accessibility permissions to control your mouse and keyboard.",
        "LSMinimumSystemVersion": "10.13",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
    },
)
