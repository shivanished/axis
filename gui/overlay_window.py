"""
Overlay window for displaying camera feed with gesture visualization.
"""
import math
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget
import cv2


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
            Qt.Tool
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

