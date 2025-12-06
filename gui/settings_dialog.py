"""
Settings dialog for configuring gesture control parameters.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout
)


class SettingsDialog(QDialog):
    """Dialog for editing application settings."""
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Camera settings
        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout()
        
        self.camera_index = QSpinBox()
        self.camera_index.setMinimum(0)
        self.camera_index.setMaximum(10)
        self.camera_index.setValue(settings.get("camera_index", 0))
        camera_layout.addRow("Camera Index:", self.camera_index)
        
        self.mirror = QCheckBox()
        self.mirror.setChecked(settings.get("mirror", True))
        camera_layout.addRow("Mirror Image:", self.mirror)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Gesture settings
        gesture_group = QGroupBox("Axis Controls")
        gesture_layout = QFormLayout()
        
        self.box_scale = QDoubleSpinBox()
        self.box_scale.setMinimum(0.2)
        self.box_scale.setMaximum(0.95)
        self.box_scale.setSingleStep(0.05)
        self.box_scale.setValue(settings.get("box_scale", 0.65))
        gesture_layout.addRow("Box Scale:", self.box_scale)
        
        self.smooth_factor = QDoubleSpinBox()
        self.smooth_factor.setMinimum(0.05)
        self.smooth_factor.setMaximum(1.0)
        self.smooth_factor.setSingleStep(0.05)
        self.smooth_factor.setValue(settings.get("smooth_factor", 0.35))
        gesture_layout.addRow("Smooth Factor:", self.smooth_factor)
        
        self.hover_threshold = QDoubleSpinBox()
        self.hover_threshold.setMinimum(0.1)
        self.hover_threshold.setMaximum(1.0)
        self.hover_threshold.setSingleStep(0.01)
        self.hover_threshold.setValue(settings.get("hover_threshold", 0.32))
        gesture_layout.addRow("Hover Threshold:", self.hover_threshold)
        
        self.click_threshold = QDoubleSpinBox()
        self.click_threshold.setMinimum(0.01)
        self.click_threshold.setMaximum(0.5)
        self.click_threshold.setSingleStep(0.01)
        self.click_threshold.setValue(settings.get("click_threshold", 0.1))
        gesture_layout.addRow("Click Threshold:", self.click_threshold)
        
        gesture_group.setLayout(gesture_layout)
        layout.addWidget(gesture_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection")
        detection_layout = QFormLayout()
        
        self.detection_confidence = QDoubleSpinBox()
        self.detection_confidence.setMinimum(0.1)
        self.detection_confidence.setMaximum(1.0)
        self.detection_confidence.setSingleStep(0.05)
        self.detection_confidence.setValue(settings.get("detection_confidence", 0.65))
        detection_layout.addRow("Detection Confidence:", self.detection_confidence)
        
        self.tracking_confidence = QDoubleSpinBox()
        self.tracking_confidence.setMinimum(0.1)
        self.tracking_confidence.setMaximum(1.0)
        self.tracking_confidence.setSingleStep(0.05)
        self.tracking_confidence.setValue(settings.get("tracking_confidence", 0.55))
        detection_layout.addRow("Tracking Confidence:", self.tracking_confidence)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def get_settings(self):
        """Get current settings from dialog."""
        return {
            "camera_index": self.camera_index.value(),
            "box_scale": self.box_scale.value(),
            "smooth_factor": self.smooth_factor.value(),
            "hover_threshold": self.hover_threshold.value(),
            "click_threshold": self.click_threshold.value(),
            "detection_confidence": self.detection_confidence.value(),
            "tracking_confidence": self.tracking_confidence.value(),
            "mirror": self.mirror.isChecked(),
        }

