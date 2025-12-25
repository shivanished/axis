"""
Main entry point for Gesture Control application.
Supports both GUI mode (default) and command-line mode (--cli flag).
"""
import argparse
import sys
from PySide6.QtWidgets import QApplication

from settings import Settings
from gui.main_window import MainWindow


def run_cli_mode(args):
    """Run in command-line mode with overlay (legacy mode)."""
    import time
    import cv2
    from PySide6.QtCore import QTimer
    from gesture_controller import GestureMouseController
    from gui.overlay_window import OverlayWindow
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to access camera index {args.camera}")

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    overlay = OverlayWindow()
    should_exit = {"exit": False}
    
    # Show overlay immediately
    overlay.show()
    
    # Handle overlay close event
    def close_event_handler(event):
        should_exit["exit"] = True
        event.accept()
    overlay.closeEvent = close_event_handler

    print("\n" + "=" * 70)
    print("GESTURE CONTROL MVP")
    print("=" * 70)
    print("• Keep your hand inside the on-screen bounding box.")
    print("• Hover the cursor by almost pinching thumb and index (leave small gap).")
    print("• Complete the pinch to trigger a click.")
    print("• Hold pinch and move hand to drag/select (like highlighting text).")
    print("• SCROLLING:")
    print("  - Show ONE finger (index) + flick = Scroll UP")
    print("  - Show TWO fingers (index+middle) + flick = Scroll DOWN")
    print("  - Slow flick (finger only) = 2 units")
    print("  - Fast flick (whole hand moves up >= 15px) = 6 units")
    print("• Close overlay window or press Cmd+Q to exit.")
    print("=" * 70 + "\n")

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(16)

    controller = GestureMouseController(
        box_scale=args.box_scale,
        smooth_factor=args.smooth,
        hover_ratio=args.hover_threshold,
        click_ratio=args.click_threshold,
        detection_confidence=args.det_conf,
        tracking_confidence=args.trk_conf,
    )

    while not should_exit["exit"]:
        app.processEvents()
        
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        processed_frame, state_info = controller.process_frame(frame, mirror=True)
        
        overlay.update_frame(processed_frame)

    cap.release()
    overlay.close()
    app.quit()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Axis Application")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode with overlay")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (CLI mode only)")
    parser.add_argument("--box-scale", type=float, default=0.65, help="Box scale (CLI mode only)")
    parser.add_argument("--smooth", type=float, default=0.35, help="Smooth factor (CLI mode only)")
    parser.add_argument("--hover-threshold", type=float, default=0.32, help="Hover threshold (CLI mode only)")
    parser.add_argument("--click-threshold", type=float, default=0.1, help="Click threshold (CLI mode only)")
    parser.add_argument("--det-conf", type=float, default=0.65, help="Detection confidence (CLI mode only)")
    parser.add_argument("--trk-conf", type=float, default=0.55, help="Tracking confidence (CLI mode only)")
    
    args = parser.parse_args()
    
    if args.cli:
        # Command-line mode (legacy)
        run_cli_mode(args)
    else:
        # GUI mode (default)
        app = QApplication(sys.argv)
        app.setApplicationName("Axis")
        
        settings = Settings()
        window = MainWindow(settings)
        window.show()
        
        sys.exit(app.exec())


if __name__ == "__main__":
    main()

