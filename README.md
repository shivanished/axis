# Gesture Control MVP

A hand gesture-based mouse controller for macOS that uses MediaPipe for hand tracking.

## Features

- **Pinch to Click**: Pinch thumb and index finger to click
- **Cursor Control**: Almost pinch to move cursor
- **Drag & Select**: Hold pinch and move to drag/select
- **Scroll Gestures**:
  - One finger flick = Scroll up
  - Two finger flick = Scroll down
- **Overlay Window**: Always-visible overlay showing camera feed with gesture visualization
- **Settings**: Configurable parameters via GUI

## Installation

### Requirements

- Python 3.11+
- macOS 10.13+
- Camera access
- Accessibility permissions (for mouse control)

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py
```

For command-line mode (legacy overlay-only mode):

```bash
python main.py --cli
```

## Building macOS App Bundle

To create a distributable `.app` bundle:

1. Install PyInstaller:

```bash
pip install pyinstaller
```

2. Run the build script:

```bash
./build_app.sh
```

Or manually:

```bash
pyinstaller gesture_control.spec --clean
```

The app bundle will be created in `dist/Gesture Control.app`.

### Permissions

After building, you may need to:

1. Grant camera permissions: System Settings > Privacy & Security > Camera
2. Grant accessibility permissions: System Settings > Privacy & Security > Accessibility
3. If the app doesn't open, right-click and select "Open" (first time only)

## Usage

### GUI Mode (Default)

1. Launch the app
2. Click "Start" to begin gesture control (overlay will appear automatically)
3. Use gestures to control your mouse

### Command-Line Mode

Run with `--cli` flag for the original overlay-only interface.

## Project Structure

```
glasses/
├── main.py                 # Entry point (supports GUI and CLI modes)
├── gesture_controller.py   # Core gesture detection logic
├── settings.py             # Settings management
├── gui/
│   ├── main_window.py      # Main GUI window
│   ├── overlay_window.py   # Overlay window
│   ├── settings_dialog.py  # Settings dialog
│   └── camera_widget.py    # Camera display widget
├── resources/              # App resources (icons, etc.)
├── gesture_control.spec    # PyInstaller spec file
└── requirements.txt        # Python dependencies
```

## Configuration

Settings are stored in `~/.gesture_control/settings.json` and can be modified via the Settings dialog in the GUI.

## Troubleshooting

- **Camera not working**: Check camera permissions in System Settings/Privacy
- **Mouse control not working**: Grant accessibility permissions
- **App won't open**: Right-click and select "Open" (macOS Gatekeeper)

## License

MIT License
