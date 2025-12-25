"""
Screen Studio-inspired bottom control panel for the Axis application.
Features collapsible UI with smooth animations and real-time stats.
"""
import time
from collections import deque
from PySide6.QtCore import (
    Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, Property
)
from PySide6.QtGui import QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QGridLayout
)

from gui.panel_styles import PANEL_STYLESHEET, COLORS


class ControlPanel(QWidget):
    """
    Bottom control panel with Screen Studio aesthetic.

    Features:
    - Recording status indicator with pulsing animation
    - Quick controls (Start, Stop, Settings buttons)
    - Real-time stats (FPS, gesture, confidence)
    - Collapsible UI with smooth animations
    """

    # Signals
    start_clicked = Signal()
    stop_clicked = Signal()
    settings_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._is_expanded = True
        self._is_tracking_active = False
        self._collapsed_height = 40
        self._expanded_height = 100

        # Gesture state tracking for debouncing
        self._last_gesture = None
        self._gesture_change_time = 0
        self._gesture_debounce_ms = 100

        # Ratio history for confidence calculation
        self._ratio_history = deque(maxlen=10)

        # Setup UI
        self.setObjectName("ControlPanel")
        self._setup_ui()
        self._setup_animations()
        self._apply_styles()

        # Set initial height
        self.setFixedHeight(self._expanded_height)

    def _setup_ui(self):
        """Setup the UI components."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Collapsed bar (hidden initially)
        self.collapsed_bar = self._create_collapsed_bar()
        self.collapsed_bar.setVisible(False)
        self.main_layout.addWidget(self.collapsed_bar)

        # Expanded panel (visible initially)
        self.expanded_panel = self._create_expanded_panel()
        self.expanded_panel.setVisible(True)
        self.main_layout.addWidget(self.expanded_panel)

    def _create_collapsed_bar(self):
        """Create the collapsed bar widget."""
        widget = QWidget()
        widget.setObjectName("CollapsedBar")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Expand button
        self.expand_button = QPushButton("∨")
        self.expand_button.setObjectName("ExpandButton")
        self.expand_button.clicked.connect(self.toggle_collapsed)
        self.expand_button.setToolTip("Expand panel")
        layout.addWidget(self.expand_button)

        # Status dot (small)
        self.status_dot_collapsed = QLabel()
        self.status_dot_collapsed.setObjectName("StatusDotCollapsedInactive")
        layout.addWidget(self.status_dot_collapsed)

        # Status text
        self.status_text_collapsed = QLabel("Inactive")
        self.status_text_collapsed.setObjectName("StatusTextCollapsed")
        layout.addWidget(self.status_text_collapsed)

        layout.addStretch()

        return widget

    def _create_expanded_panel(self):
        """Create the expanded panel widget."""
        widget = QWidget()
        widget.setObjectName("ExpandedPanel")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(20)

        # Left section: Status + Controls
        left_section = self._create_controls_section()
        layout.addWidget(left_section)

        # Separator
        separator1 = QFrame()
        separator1.setObjectName("Separator")
        separator1.setFrameShape(QFrame.VLine)
        layout.addWidget(separator1)

        # Stats section
        stats_section = self._create_stats_section()
        layout.addWidget(stats_section)

        # Separator
        separator2 = QFrame()
        separator2.setObjectName("Separator")
        separator2.setFrameShape(QFrame.VLine)
        layout.addWidget(separator2)

        # Right section: Collapse button
        right_section = self._create_right_section()
        layout.addWidget(right_section)

        return widget

    def _create_controls_section(self):
        """Create the controls section (status + buttons)."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Status indicator
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self.status_dot = QLabel()
        self.status_dot.setObjectName("StatusDotInactive")
        status_layout.addWidget(self.status_dot)

        self.status_text = QLabel("Inactive")
        self.status_text.setObjectName("StatusText")
        status_layout.addWidget(self.status_text)

        layout.addWidget(status_widget)

        # Buttons
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_clicked.emit)
        self.start_button.setToolTip("Start gesture tracking")
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop gesture tracking")
        layout.addWidget(self.stop_button)

        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.clicked.connect(self.settings_clicked.emit)
        self.settings_button.setToolTip("Open settings")
        layout.addWidget(self.settings_button)

        return widget

    def _create_stats_section(self):
        """Create the stats display section."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # FPS
        fps_widget = QWidget()
        fps_layout = QVBoxLayout(fps_widget)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.setSpacing(2)

        fps_label = QLabel("FPS")
        fps_label.setObjectName("StatLabel")
        fps_layout.addWidget(fps_label)

        self.fps_value = QLabel("--")
        self.fps_value.setObjectName("StatValueMono")
        self.fps_value.setToolTip("Frames per second")
        fps_layout.addWidget(self.fps_value)

        layout.addWidget(fps_widget)

        # Gesture
        gesture_widget = QWidget()
        gesture_layout = QVBoxLayout(gesture_widget)
        gesture_layout.setContentsMargins(0, 0, 0, 0)
        gesture_layout.setSpacing(2)

        gesture_label = QLabel("GESTURE")
        gesture_label.setObjectName("StatLabel")
        gesture_layout.addWidget(gesture_label)

        self.gesture_value = QLabel("Ready")
        self.gesture_value.setObjectName("StatValue")
        self.gesture_value.setToolTip("Current gesture state")
        gesture_layout.addWidget(self.gesture_value)

        layout.addWidget(gesture_widget)

        # Confidence
        confidence_widget = QWidget()
        confidence_layout = QVBoxLayout(confidence_widget)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(2)

        confidence_label = QLabel("CONFIDENCE")
        confidence_label.setObjectName("StatLabel")
        confidence_layout.addWidget(confidence_label)

        self.confidence_value = QLabel("--")
        self.confidence_value.setObjectName("StatValueMono")
        self.confidence_value.setToolTip("Tracking confidence")
        confidence_layout.addWidget(self.confidence_value)

        layout.addWidget(confidence_widget)

        return widget

    def _create_right_section(self):
        """Create the right section with collapse button."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.collapse_button = QPushButton("∧")
        self.collapse_button.setObjectName("CollapseButton")
        self.collapse_button.clicked.connect(self.toggle_collapsed)
        self.collapse_button.setToolTip("Collapse panel")
        layout.addWidget(self.collapse_button)

        return widget

    def _setup_animations(self):
        """Setup animations for collapse/expand."""
        # Opacity effect for expanded panel
        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(1.0)
        self.expanded_panel.setGraphicsEffect(self.opacity_effect)

        # Animation group
        self.animation_group = QParallelAnimationGroup()

        # Height animation
        self.height_animation = QPropertyAnimation(self, b"maximumHeight")
        self.height_animation.setDuration(300)
        self.height_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation_group.addAnimation(self.height_animation)

        # Minimum height animation
        self.min_height_animation = QPropertyAnimation(self, b"minimumHeight")
        self.min_height_animation.setDuration(300)
        self.min_height_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation_group.addAnimation(self.min_height_animation)

        # Opacity animation
        self.opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(250)
        self.opacity_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation_group.addAnimation(self.opacity_animation)

        # Pulsing animation for status dot
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse_status_dot)
        self._pulse_direction = 1  # 1 for fading out, -1 for fading in
        self._pulse_opacity = 1.0

    def _apply_styles(self):
        """Apply stylesheets and visual effects."""
        self.setStyleSheet(PANEL_STYLESHEET)

        # Add drop shadow to panel
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, -2)
        self.setGraphicsEffect(shadow)

    def toggle_collapsed(self):
        """Toggle between collapsed and expanded states."""
        if self._is_expanded:
            self._collapse()
        else:
            self._expand()

    def _collapse(self):
        """Collapse the panel to minimal bar."""
        self._is_expanded = False

        # Setup animations
        self.height_animation.setStartValue(self._expanded_height)
        self.height_animation.setEndValue(self._collapsed_height)

        self.min_height_animation.setStartValue(self._expanded_height)
        self.min_height_animation.setEndValue(self._collapsed_height)

        self.opacity_animation.setStartValue(1.0)
        self.opacity_animation.setEndValue(0.0)

        # Start animation
        self.animation_group.start()

        # Switch visibility after animation
        QTimer.singleShot(300, self._finalize_collapse)

    def _expand(self):
        """Expand the panel to full view."""
        self._is_expanded = True

        # Switch visibility immediately
        self.expanded_panel.setVisible(True)
        self.collapsed_bar.setVisible(False)

        # Setup animations
        self.height_animation.setStartValue(self._collapsed_height)
        self.height_animation.setEndValue(self._expanded_height)

        self.min_height_animation.setStartValue(self._collapsed_height)
        self.min_height_animation.setEndValue(self._expanded_height)

        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)

        # Start animation
        self.animation_group.start()

    def _finalize_collapse(self):
        """Finalize collapse animation."""
        if not self._is_expanded:
            self.expanded_panel.setVisible(False)
            self.collapsed_bar.setVisible(True)

    def _pulse_status_dot(self):
        """Pulse the status dot when tracking is active."""
        if not self._is_tracking_active:
            return

        # Adjust opacity
        self._pulse_opacity += self._pulse_direction * 0.05

        # Reverse direction at limits
        if self._pulse_opacity >= 1.0:
            self._pulse_opacity = 1.0
            self._pulse_direction = -1
        elif self._pulse_opacity <= 0.3:
            self._pulse_opacity = 0.3
            self._pulse_direction = 1

        # Apply opacity to status dot
        self.status_dot.setStyleSheet(
            f"background-color: {COLORS['accent_red']}; "
            f"border-radius: 6px; "
            f"min-width: 12px; max-width: 12px; "
            f"min-height: 12px; max-height: 12px; "
            f"opacity: {self._pulse_opacity};"
        )

    def set_tracking_active(self, active: bool):
        """Update the tracking status indicator."""
        self._is_tracking_active = active

        if active:
            # Active state
            self.status_dot.setObjectName("StatusDot")
            self.status_dot_collapsed.setObjectName("StatusDotCollapsed")
            self.status_text.setText("Tracking Active")
            self.status_text_collapsed.setText("Active")

            # Enable stop button, disable start button
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            # Start pulsing animation
            self.pulse_timer.start(50)  # 20fps for pulse
        else:
            # Inactive state
            self.status_dot.setObjectName("StatusDotInactive")
            self.status_dot_collapsed.setObjectName("StatusDotCollapsedInactive")
            self.status_text.setText("Inactive")
            self.status_text_collapsed.setText("Inactive")

            # Enable start button, disable stop button
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            # Stop pulsing animation
            self.pulse_timer.stop()
            self._pulse_opacity = 1.0
            self._pulse_direction = 1

        # Reapply stylesheet to update colors
        self.setStyleSheet(PANEL_STYLESHEET)

    def update_stats(self, state_info: dict, fps: float):
        """
        Update real-time stats display.

        Args:
            state_info: Dict containing gesture state information
            fps: Current frames per second
        """
        if not state_info:
            return

        # Update FPS
        self._update_fps(fps)

        # Update gesture (with debouncing)
        self._update_gesture(state_info.get('label', 'unknown'))

        # Update confidence
        self._update_confidence(state_info.get('ratio', 0.0))

    def _update_fps(self, fps: float):
        """Update FPS display with color coding."""
        fps_int = int(round(fps))
        self.fps_value.setText(str(fps_int))

        # Color coding
        if fps_int >= 25:
            self.fps_value.setObjectName("FPSGood")
        elif fps_int >= 15:
            self.fps_value.setObjectName("FPSMedium")
        else:
            self.fps_value.setObjectName("FPSPoor")
            self.fps_value.setToolTip("Low FPS - check system performance")

        # Reapply stylesheet
        self.setStyleSheet(PANEL_STYLESHEET)

    def _update_gesture(self, label: str):
        """Update gesture display with debouncing."""
        current_time = time.time() * 1000  # Convert to ms

        # Check if gesture has changed
        if label != self._last_gesture:
            self._last_gesture = label
            self._gesture_change_time = current_time
            return  # Don't update yet, wait for debounce

        # Check if enough time has passed since last change
        if current_time - self._gesture_change_time < self._gesture_debounce_ms:
            return

        # Map label to user-friendly text
        gesture_map = {
            'no-hand': 'Waiting for hand...',
            'outside': 'Move hand to box',
            'open': 'Hand open',
            'ready': 'Ready to click',
            'click': 'Clicking',
            'scroll-one': 'Scroll (1 finger)',
            'scroll-two': 'Scroll (2 fingers)',
        }

        gesture_text = gesture_map.get(label, 'Unknown')
        self.gesture_value.setText(gesture_text)

    def _update_confidence(self, ratio: float):
        """Update confidence display based on ratio stability."""
        # Add to history
        self._ratio_history.append(ratio)

        if len(self._ratio_history) < 3:
            self.confidence_value.setText("--")
            return

        # Calculate variance as stability metric
        mean_ratio = sum(self._ratio_history) / len(self._ratio_history)
        variance = sum((x - mean_ratio) ** 2 for x in self._ratio_history) / len(self._ratio_history)

        # Map variance to confidence (lower variance = higher confidence)
        if variance < 0.005:
            confidence = 95
        elif variance < 0.01:
            confidence = 90
        elif variance < 0.02:
            confidence = 85
        elif variance < 0.05:
            confidence = 75
        elif variance < 0.1:
            confidence = 65
        else:
            confidence = 50

        self.confidence_value.setText(f"{confidence}%")
