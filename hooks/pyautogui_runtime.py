"""
Runtime hook to ensure PyAutoGUI works correctly in PyInstaller bundle on macOS.
This hook runs at runtime, not during build analysis.
"""
import sys
import os

# Only run at runtime, not during PyInstaller analysis
if sys.platform == 'darwin' and not hasattr(sys, 'frozen'):
    # This shouldn't run during PyInstaller analysis, but just in case:
    pass
elif sys.platform == 'darwin':
    # Runtime: Ensure PyAutoGUI can find its dependencies
    try:
        import pyautogui
        # Test PyAutoGUI functionality
        try:
            # Try to get screen size - this will fail if Quartz isn't accessible
            size = pyautogui.size()
            if size is None or size[0] == 0 or size[1] == 0:
                raise RuntimeError("PyAutoGUI.size() returned invalid values")
            
            # Try to get mouse position - this requires Accessibility permissions
            pos = pyautogui.position()
            if pos is None:
                raise RuntimeError("PyAutoGUI.position() returned None - check Accessibility permissions")
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_msg = (
                f"PyAutoGUI test failed: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                "This usually means:\n"
                "1. Accessibility permissions not granted\n"
                "2. Quartz framework not accessible in bundle\n"
                "3. App needs to be code-signed\n"
            )
            # Store error for later display
            os.environ['PYAUTOGUI_ERROR'] = error_msg
    except ImportError as e:
        os.environ['PYAUTOGUI_ERROR'] = f"PyAutoGUI import failed: {e}"

