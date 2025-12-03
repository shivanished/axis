"""
Overlay window for displaying camera feed with gesture visualization.
"""
import math
import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget
import cv2

# macOS-specific imports for non-activating floating panel
if sys.platform == 'darwin':
    try:
        from AppKit import (
            NSWindow, NSPanel,
            NSFloatingWindowLevel, NSStatusWindowLevel,
            NSNonactivatingPanelMask, NSBorderlessWindowMask,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorStationary
        )
        import objc
        MACOS_AVAILABLE = True
    except ImportError:
        NSWindow = None
        NSPanel = None
        MACOS_AVAILABLE = False
else:
    MACOS_AVAILABLE = False


class OverlayWindow(QWidget):
    """
    Frameless, always-on-top overlay window that displays the camera feed.
    Can be dragged and snaps to screen corners.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowDoesNotAcceptFocus  # Don't steal focus from other apps
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # Window properties
        self.overlay_width = 400
        self.overlay_height = 300
        self.setFixedSize(self.overlay_width, self.overlay_height)
        
        # Dragging state
        self._dragging = False
        self._drag_position = None
        
        # Corner snapping threshold
        self._snap_threshold = 50
        
        # Get screen dimensions
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        
        # Position in top-right corner by default
        self._snap_to_corner("top-right")
        
        # Create label for displaying frames
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.label.setGeometry(0, 0, self.overlay_width, self.overlay_height)
        
        # Set window title for identification
        self.setWindowTitle("Gesture Control Overlay")
        
        # Store reference to NSWindow when we find it (for macOS)
        self._ns_window = None
        self._window_configured = False
    
    def showEvent(self, event):
        """Handle show event to configure non-activating floating panel."""
        super().showEvent(event)
        if not self._window_configured:
            # Use a delayed call to ensure window is fully initialized
            QTimer.singleShot(100, self._configure_non_activating_panel)
    
    def _configure_non_activating_panel(self):
        """Configure the window as a non-activating floating panel on macOS."""
        if not MACOS_AVAILABLE or self._window_configured:
            return
        
        try:
            # Get the QWindow handle
            qwindow = self.windowHandle()
            if not qwindow:
                # Retry after a short delay if window handle not ready
                QTimer.singleShot(100, self._configure_non_activating_panel)
                return
            
            # Access NSWindow through PyObjC
            from AppKit import NSApp
            
            # Find our window by matching properties
            if self._ns_window is None:
                self._ns_window = self._find_native_window(qwindow, NSApp)
            
            # Configure as non-activating floating panel
            if self._ns_window:
                self._apply_non_activating_config()
                self._window_configured = True
        except Exception as e:
            # Log but don't crash - fall back to Qt's behavior
            print(f"Warning: Could not configure non-activating panel: {e}")
    
    def _find_native_window(self, qwindow, NSApp):
        """Find the native NSWindow corresponding to this QWidget."""
        try:
            # Method 1: Try direct access through QWindow's native handle
            try:
                win_id = qwindow.winId()
                if win_id:
                    # Try to get NSWindow directly from the window handle
                    # PySide6 on macOS may provide direct access
                    for window in NSApp.windows():
                        try:
                            # Match by checking if it's the most recently created visible window
                            # that matches our properties
                            if window.isVisible():
                                frame = window.frame()
                                if (abs(frame.size.width - self.width()) < 10 and
                                    abs(frame.size.height - self.height()) < 10):
                                    # Check if frameless (borderless)
                                    style_mask = window.styleMask()
                                    # NSBorderlessWindowMask = 0, or check for lack of title bar
                                    if style_mask & 0x1 == 0:  # No title bar mask
                                        return window
                        except Exception:
                            continue
            except Exception:
                pass
            
            # Method 2: Match by title
            target_title = self.windowTitle()
            for window in NSApp.windows():
                try:
                    if window.isVisible() and hasattr(window, 'title'):
                        if window.title() == target_title:
                            return window
                except Exception:
                    continue
            
            # Method 3: Match by size and position (last resort)
            target_width = self.width()
            target_height = self.height()
            target_x = self.x()
            target_y = self.y()
            
            for window in NSApp.windows():
                try:
                    if not window.isVisible():
                        continue
                    frame = window.frame()
                    if (abs(frame.size.width - target_width) < 10 and
                        abs(frame.size.height - target_height) < 10 and
                        abs(frame.origin.x - target_x) < 10 and
                        abs(frame.origin.y - target_y) < 10):
                        return window
                except Exception:
                    continue
        except Exception:
            pass
        
        return None
    
    def _apply_non_activating_config(self):
        """Apply non-activating floating panel configuration to NSWindow."""
        if not self._ns_window:
            return
        
        try:
            # Set window level to floating (above normal windows)
            self._ns_window.setLevel_(NSFloatingWindowLevel)
            
            # Get current style mask
            current_mask = self._ns_window.styleMask()
            
            # Add non-activating panel mask
            # NSNonactivatingPanelMask = 0x20
            new_mask = current_mask | NSNonactivatingPanelMask
            
            # Apply the new style mask
            self._ns_window.setStyleMask_(new_mask)
            
            # Don't hide when app deactivates (stays visible when switching apps)
            self._ns_window.setHidesOnDeactivate_(False)
            
            # Set collection behavior to stay on all spaces and be stationary
            collection_behavior = (
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )
            self._ns_window.setCollectionBehavior_(collection_behavior)
            
            # Ensure window cannot become key window (won't activate app)
            # For NSPanel, this is automatic, but we ensure it for NSWindow too
            if hasattr(self._ns_window, 'setCanBecomeKeyWindow_'):
                # This method doesn't exist directly, but we can override behavior
                # The non-activating panel mask should handle this
                pass
            
            # Bring to front without activating
            self._ns_window.orderFront_(None)
            
        except Exception as e:
            print(f"Warning: Could not apply all non-activating config: {e}")
            # Try at least setting the level
            try:
                self._ns_window.setLevel_(NSFloatingWindowLevel)
                self._ns_window.setHidesOnDeactivate_(False)
            except Exception:
                pass
        
    def _snap_to_corner(self, corner):
        """Snap window to specified corner."""
        if corner == "top-left":
            x = 0
            y = 0
        elif corner == "top-right":
            x = self.screen_width - self.overlay_width
            y = 0
        elif corner == "bottom-left":
            x = 0
            y = self.screen_height - self.overlay_height
        elif corner == "bottom-right":
            x = self.screen_width - self.overlay_width
            y = self.screen_height - self.overlay_height
        else:
            return
        
        self.move(x, y)
    
    def _get_nearest_corner(self, x, y):
        """Determine nearest corner based on position."""
        corners = {
            "top-left": (0, 0),
            "top-right": (self.screen_width - self.overlay_width, 0),
            "bottom-left": (0, self.screen_height - self.overlay_height),
            "bottom-right": (self.screen_width - self.overlay_width, self.screen_height - self.overlay_height),
        }
        
        min_dist = float('inf')
        nearest = "top-right"
        
        for corner_name, (cx, cy) in corners.items():
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = corner_name
        
        # Only snap if within threshold
        if min_dist <= self._snap_threshold:
            return nearest
        return None
    
    def mousePressEvent(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if self._dragging and event.buttons() == Qt.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_position
            self.move(new_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release and snap to corner."""
        if event.button() == Qt.LeftButton:
            self._dragging = False
            
            # Get current window position
            current_x = self.x()
            current_y = self.y()
            
            # Find nearest corner and snap if within threshold
            nearest = self._get_nearest_corner(current_x, current_y)
            if nearest:
                self._snap_to_corner(nearest)
            
            event.accept()
    
    def update_frame(self, frame):
        """Update the displayed frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit overlay while maintaining aspect ratio
        h, w = rgb_frame.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > self.overlay_width / self.overlay_height:
            # Frame is wider
            new_width = self.overlay_width
            new_height = int(self.overlay_width / aspect_ratio)
        else:
            # Frame is taller
            new_height = self.overlay_height
            new_width = int(self.overlay_height * aspect_ratio)
        
        resized = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to QImage
        h, w, ch = resized.shape
        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap)

