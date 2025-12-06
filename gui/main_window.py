"""
Main window for the gesture control application.
"""
import cv2
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
    QPushButton, QMenuBar, QStatusBar, QMessageBox
)

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from gesture_controller import GestureMouseController
from gui.overlay_window import OverlayWindow
from gui.settings_dialog import SettingsDialog


class FrameProcessor(QObject):
    """Worker object for processing frames in a separate thread."""
    frame_processed = Signal(object, object)  # frame, state_info
    
    def __init__(self, controller, mirror=True):
        super().__init__()
        self.controller = controller
        self.mirror = mirror
        self.running = False
    
    def process_frame(self, frame):
        """Process a frame and emit signal."""
        if self.running:
            processed_frame, state_info = self.controller.process_frame(frame, mirror=self.mirror)
            self.frame_processed.emit(processed_frame, state_info)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Axis")
        self.setMinimumSize(400, 300)
        
        # Initialize controller
        self.controller = None
        self.cap = None
        self.running = False
        
        # Overlay window (always visible when running)
        self.overlay = OverlayWindow()
        self.overlay_visible = True  # Always show overlay
        
        # Frame processor
        self.frame_processor = None
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        
        # Timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
    
    def _setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Axis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;")
        layout.addWidget(title)
        
        # Status label
        self.status_label = QLabel("Not running")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_control)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_control)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self._show_settings)
        button_layout.addWidget(self.settings_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        settings_action = file_menu.addAction("Settings...")
        settings_action.triggered.connect(self._show_settings)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)
    
    def _start_control(self):
        """Start gesture control."""
        camera_index = self.settings.get("camera_index", 0)
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", f"Could not open camera {camera_index}")
                return
            
            # Create controller with current settings
            self.controller = GestureMouseController(
                box_scale=self.settings.get("box_scale", 0.65),
                smooth_factor=self.settings.get("smooth_factor", 0.35),
                hover_ratio=self.settings.get("hover_threshold", 0.32),
                click_ratio=self.settings.get("click_threshold", 0.1),
                detection_confidence=self.settings.get("detection_confidence", 0.65),
                tracking_confidence=self.settings.get("tracking_confidence", 0.55),
            )
            
            self.running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Running")
            self.statusBar().showMessage("Gesture control active")
            
            # Show overlay window
            self.overlay.show()
            self.overlay_visible = True
            
            # Start timer for frame updates
            self.timer.start(33)  # ~30fps
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
    
    def _stop_control(self):
        """Stop gesture control."""
        self.running = False
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.controller = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopped")
        self.statusBar().showMessage("Gesture control stopped")
        
        # Hide overlay when stopped
        self.overlay.hide()
        self.overlay_visible = False
    
    def _update_frame(self):
        """Update frame from camera."""
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        if self.controller:
            processed_frame, state_info = self.controller.process_frame(
                frame,
                mirror=self.settings.get("mirror", True)
            )
            
            # Always update overlay when running
            self.overlay.update_frame(processed_frame)
    
    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.settings.update(**new_settings)
            
            # If running, restart with new settings
            if self.running:
                self._stop_control()
                self._start_control()
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Axis",
            "Axis\n\n"
            "Control your mouse with hand gestures.\n\n"
            "• Pinch to click\n"
            "• Almost pinch to move cursor\n"
            "• One finger flick to scroll up\n"
            "• Two finger flick to scroll down"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        self._stop_control()
        self.overlay.close()
        event.accept()

