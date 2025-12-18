"""
Settings dialog for configuring gesture control parameters.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout, QSlider
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

        # Smoothing settings (One Euro Filter)
        smoothing_group = QGroupBox("Cursor Smoothing")
        smoothing_layout = QFormLayout()

        self.use_one_euro = QCheckBox()
        self.use_one_euro.setChecked(settings.get("use_one_euro", True))
        self.use_one_euro.toggled.connect(self._on_smoothing_toggled)
        smoothing_layout.addRow("Enable Advanced Smoothing:", self.use_one_euro)

        # Smoothness slider (0-100 mapped to 0.0-1.0)
        smoothness_container = QVBoxLayout()
        self.smoothness_slider = QSlider(Qt.Horizontal)
        self.smoothness_slider.setMinimum(0)
        self.smoothness_slider.setMaximum(100)
        smoothness_value = int(settings.get("one_euro_smoothness", 0.5) * 100)
        self.smoothness_slider.setValue(smoothness_value)
        self.smoothness_slider.setEnabled(self.use_one_euro.isChecked())
        self.smoothness_slider.valueChanged.connect(self._on_smoothness_changed)

        # Labels for slider
        slider_labels = QHBoxLayout()
        slider_labels.addWidget(QLabel("More Smoothing"))
        slider_labels.addStretch()
        slider_labels.addWidget(QLabel("More Responsive"))

        smoothness_container.addWidget(self.smoothness_slider)
        smoothness_container.addLayout(slider_labels)

        self.smoothness_value_label = QLabel(f"Value: {smoothness_value / 100:.2f}")
        smoothing_layout.addRow("Smoothness/Responsiveness:", smoothness_container)
        smoothing_layout.addRow("", self.smoothness_value_label)

        # Frame interpolation toggle
        self.use_interpolation = QCheckBox()
        self.use_interpolation.setChecked(settings.get("use_interpolation", False))
        smoothing_layout.addRow("Enable Frame Interpolation:", self.use_interpolation)

        interpolation_help = QLabel("(Adds latency but creates smoother motion)")
        interpolation_help.setStyleSheet("color: gray; font-size: 10px;")
        smoothing_layout.addRow("", interpolation_help)

        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)

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

    def _on_smoothing_toggled(self, checked):
        """Handle smoothing toggle."""
        self.smoothness_slider.setEnabled(checked)

    def _on_smoothness_changed(self, value):
        """Handle smoothness slider change."""
        smoothness = value / 100.0
        self.smoothness_value_label.setText(f"Value: {smoothness:.2f}")

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
            "use_one_euro": self.use_one_euro.isChecked(),
            "one_euro_smoothness": self.smoothness_slider.value() / 100.0,
            "use_interpolation": self.use_interpolation.isChecked(),
        }

