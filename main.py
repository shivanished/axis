import argparse
import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput import keyboard
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget


class SimpleHandTracker:
    """
    Lightweight helper around MediaPipe that returns pixel-space landmarks
    for a single hand. The tracker draws landmarks onto the provided frame
    so that the user can see what the camera is detecting.
    """

    def __init__(self, detection_confidence=0.65, tracking_confidence=0.55):
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def locate(self, frame, draw=True):
        """
        Runs hand tracking on the provided frame and returns pixel-space
        landmarks plus the bounding box surrounding the detected hand.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        points = []
        for lm in hand_landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        if draw:
            self._drawer.draw_landmarks(frame, hand_landmarks, self._connections)

        return {"points": points, "bbox": bbox}


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


class GestureMouseController:
    """
    Tracks a single hand inside a visual bounding box. The midpoint between
    the thumb tip and index finger tip determines the on-screen cursor
    position whenever the user is in the "almost pinched" state. A full
    pinch triggers an automatic click.
    """

    def __init__(
        self,
        box_scale=0.65,
        smooth_factor=0.35,
        hover_ratio=0.32,
        click_ratio=0.1,
        detection_confidence=0.65,
        tracking_confidence=0.55,
    ):
        if not 0.2 <= box_scale <= 0.95:
            raise ValueError("box_scale must be inside [0.2, 0.95]")
        if not 0.05 <= smooth_factor <= 1.0:
            raise ValueError("smooth_factor must be inside [0.05, 1.0]")
        if hover_ratio <= click_ratio:
            raise ValueError("hover_ratio must be larger than click_ratio")

        self.box_scale = box_scale
        self.smooth_factor = smooth_factor
        self.hover_ratio = hover_ratio
        self.click_ratio = click_ratio

        self.detector = SimpleHandTracker(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )

        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        start_x, start_y = pyautogui.position()
        self.cursor_x = start_x
        self.cursor_y = start_y

        self._click_latched = False
        self._drag_active = False  # Track if we're currently dragging
        self._drag_start_pos = None  # Position where drag started
        self._pinch_start_pos = None  # Position when pinch started
        self._last_state = "no-hand"
        self._last_ratio = None
        
        # Scrolling state
        self._hand_position_history = []  # Track finger and wrist positions [(finger_x, finger_y, wrist_x, wrist_y, timestamp), ...]
        self._max_history = 15  # Keep last 15 positions
        self._flick_threshold = 30  # Minimum pixel movement for flick detection
        self._last_scroll_time = time.time()
        self._scroll_cooldown = 0.2  # Minimum time between scroll triggers (seconds)
        self._wrist_movement_threshold = 15  # Pixels - if wrist moves up this much, it's a fast flick
        self._slow_scroll_amount = 2  # Small scroll for slow flicks
        self._fast_scroll_amount = 6  # More scroll for fast flicks

    def run(self, camera_index=0, mirror=True):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to access camera index {camera_index}")

        # Initialize Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create overlay window
        overlay = OverlayWindow()
        overlay_visible = False  # Start hidden
        
        # Global keyboard listener state
        overlay_visible_state = {"visible": False}
        should_exit = {"exit": False}
        
        def toggle_overlay():
            """Toggle overlay visibility - must be called on main thread."""
            overlay_visible_state["visible"] = not overlay_visible_state["visible"]
            if overlay_visible_state["visible"]:
                overlay.show()
            else:
                overlay.hide()
        
        def toggle_overlay_safe():
            """Schedule toggle_overlay to run on main thread."""
            QTimer.singleShot(0, toggle_overlay)
        
        # Set up global keyboard listener for Cmd+I
        def setup_hotkey_listener():
            """Set up global hotkey listener in separate thread."""
            try:
                # Use HotKey for better macOS support
                # Try different command key representations for macOS
                hotkey_combos = [
                    '<cmd>+i',  # Standard format
                    '<cmd_l>+i',
                    '<cmd_r>+i',
                ]
                
                hotkey = None
                for combo in hotkey_combos:
                    try:
                        hotkey = keyboard.HotKey(
                            keyboard.HotKey.parse(combo),
                            toggle_overlay_safe
                        )
                        break
                    except:
                        continue
                
                if hotkey is None:
                    # Fallback to manual detection
                    pressed_keys = set()
                    
                    def on_press(key):
                        try:
                            if hasattr(key, 'char') and key.char and key.char.lower() == 'i':
                                if any(k in pressed_keys for k in [keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r]):
                                    toggle_overlay_safe()
                            elif key in [keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r]:
                                pressed_keys.add(key)
                        except:
                            pass
                    
                    def on_release(key):
                        try:
                            if key in [keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r]:
                                pressed_keys.discard(key)
                        except:
                            pass
                    
                    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                        listener.join()
                else:
                    # Use HotKey listener
                    def on_press(key):
                        try:
                            hotkey.press(key)
                        except:
                            pass
                    
                    def on_release(key):
                        try:
                            hotkey.release(key)
                        except:
                            pass
                    
                    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                        listener.join()
            except Exception as e:
                print(f"Warning: Could not set up global hotkey listener: {e}")
                print("You may need to grant Accessibility permissions in System Settings.")
                print("On macOS: System Settings > Privacy & Security > Accessibility")
        
        # Start keyboard listener in background thread
        keyboard_thread = threading.Thread(target=setup_hotkey_listener, daemon=True)
        keyboard_thread.start()
        
        # Store exit flag reference in overlay for close event
        overlay._should_exit = should_exit
        
        # Override closeEvent method
        original_close_event = overlay.closeEvent
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
        print("• Press Cmd+I to toggle overlay visibility.")
        print("• Press Cmd+Q or close overlay to exit.")
        print("=" * 70 + "\n")

        # Timer for processing Qt events
        timer = QTimer()
        timer.timeout.connect(lambda: None)  # Just process events
        timer.start(16)  # ~60fps

        overlay_visible = False  # Track current visibility state

        while not should_exit["exit"]:
            # Process Qt events
            app.processEvents()
            
            # Check if overlay visibility changed
            if overlay_visible_state["visible"] != overlay_visible:
                overlay_visible = overlay_visible_state["visible"]
                if overlay_visible:
                    overlay.show()
                else:
                    overlay.hide()
            
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, attempting to continue...")
                time.sleep(0.01)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]
            control_box = self._control_box(frame_width, frame_height)

            hand_info = self.detector.locate(frame, draw=True)
            if hand_info:
                state_info = self._handle_hand(frame, hand_info, control_box)
            else:
                self._click_latched = False
                # Clear hand position history when hand is lost
                self._hand_position_history = []
                # Release drag if active
                if self._drag_active:
                    pyautogui.mouseUp()
                    self._drag_active = False
                self._drag_start_pos = None
                self._pinch_start_pos = None
                state_info = {
                    "label": "no-hand",
                    "ratio": None,
                    "midpoint": None,
                    "inside_box": False,
                    "finger_gesture": None,
                    "flick_direction": None,
                }

            self._draw_guides(frame, control_box, state_info)

            # Update overlay if visible
            if overlay_visible:
                overlay.update_frame(frame)

        cap.release()
        overlay.close()
        app.quit()

    def _control_box(self, frame_width, frame_height):
        box_w = int(frame_width * self.box_scale)
        box_h = int(frame_height * self.box_scale)
        left = (frame_width - box_w) // 2
        top = (frame_height - box_h) // 2
        right = left + box_w
        bottom = top + box_h
        return left, top, right, bottom

    def _handle_hand(self, frame, hand_info, control_box):
        points = hand_info["points"]
        thumb = points[4]
        index = points[8]
        midpoint = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        bbox = hand_info["bbox"]

        pinch_distance = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
        hand_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1)
        pinch_ratio = pinch_distance / hand_size

        # Detect finger gesture for scrolling
        finger_gesture = self._detect_finger_gesture(points)
        
        # Get wrist position (landmark 0)
        wrist = points[0]
        
        # Track finger and wrist positions for flick detection
        current_time = time.time()
        self._hand_position_history.append((midpoint[0], midpoint[1], wrist[0], wrist[1], current_time))
        if len(self._hand_position_history) > self._max_history:
            self._hand_position_history.pop(0)
        
        # Detect flick (based on finger movement)
        flick_direction = self._detect_flick(midpoint)
        
        # Calculate wrist movement for fast/slow detection
        wrist_movement = None
        if finger_gesture and flick_direction and len(self._hand_position_history) >= 2:
            window_size = min(5, len(self._hand_position_history))
            recent_window = self._hand_position_history[-window_size:]
            if len(recent_window) >= 2:
                start_entry = recent_window[0]
                end_entry = recent_window[-1]
                # Wrist movement: negative means moved up (y decreases)
                wrist_dy = start_entry[3] - end_entry[3]  # Start y - end y (positive = moved up)
                wrist_movement = wrist_dy
        
        # Perform discrete scroll if gesture detected
        if finger_gesture and flick_direction:
            self._perform_scroll(finger_gesture, flick_direction, wrist_movement)
            # Reset history after detecting flick to avoid continuous scrolling
            current_time = time.time()
            self._hand_position_history = [(midpoint[0], midpoint[1], wrist[0], wrist[1], current_time)]

        left, top, right, bottom = control_box
        inside_box = left <= midpoint[0] <= right and top <= midpoint[1] <= bottom

        if not inside_box:
            state_label = "outside"
            self._click_latched = False
            screen_target = None
            # Release drag if active
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            self._drag_start_pos = None
            self._pinch_start_pos = None
        else:
            # Prioritize pinch detection over scroll gestures
            # If pinching (even slightly), handle pinch/click, ignore scroll gestures
            if pinch_ratio < self.click_ratio:
                state_label = "click"
            elif pinch_ratio < self.hover_ratio:
                state_label = "ready"
            # Only check for scroll gestures if NOT pinching
            elif finger_gesture:
                state_label = f"scroll-{finger_gesture}"
                screen_target = None
                self._click_latched = False
            else:
                state_label = "open"

            if state_label in ("ready", "click"):
                screen_target = self._map_to_screen(midpoint, control_box)
                self._move_cursor(screen_target)
                
                # Handle click and drag functionality
                if state_label == "click":
                    if not self._click_latched:
                        # New pinch - click immediately (first priority)
                        pyautogui.click()
                        self._click_latched = True
                        self._pinch_start_pos = screen_target
                        self._drag_start_pos = screen_target
                    else:
                        # Pinch is held - check if hand moved (start drag)
                        if self._pinch_start_pos is not None:
                            # Check if hand moved significantly from where pinch started
                            move_distance = math.hypot(
                                screen_target[0] - self._pinch_start_pos[0],
                                screen_target[1] - self._pinch_start_pos[1]
                            )
                            if move_distance > 10:  # Moved more than 10 pixels
                                # Start dragging (hold mouse button down)
                                if not self._drag_active:
                                    self._drag_active = True
                                    pyautogui.mouseDown()
                                    # Update drag start to current position to avoid jump
                                    self._drag_start_pos = screen_target
                    
                    # If dragging, continue moving cursor (mouse is already down)
                    if self._drag_active:
                        # Cursor is already moved above, mouse button is held down
                        pass
                else:
                    # In "ready" state - not clicking, so no drag
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    self._drag_start_pos = None
                    self._pinch_start_pos = None
            else:
                screen_target = None
                if state_label not in ("scroll-one", "scroll-two"):
                    self._click_latched = False
                    # Release drag if active
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    self._drag_start_pos = None
                    self._pinch_start_pos = None

            # Reset click latch when leaving click state
            if state_label != "click":
                if state_label not in ("scroll-one", "scroll-two"):
                    # If we were dragging, release mouse button
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    # Click already happened when entering click state, so just reset
                    self._click_latched = False
                    self._pinch_start_pos = None

        # Visual cues for thumb/index/midpoint
        cv2.circle(frame, thumb, 8, (0, 165, 255), -1)
        cv2.circle(frame, index, 8, (255, 90, 120), -1)
        cv2.line(frame, thumb, index, (255, 255, 255), 2)
        
        # Highlight differently when dragging
        if self._drag_active:
            cv2.circle(frame, midpoint, 15, (0, 0, 255), 3)  # Red circle when dragging
            cv2.putText(frame, "DRAGGING", (midpoint[0] - 40, midpoint[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(frame, midpoint, 10, (0, 255, 0), 2)  # Green circle normally
        
        # Visual feedback for scrolling gestures
        if finger_gesture == "one":
            # Highlight index finger
            cv2.circle(frame, index, 15, (0, 255, 255), 3)
            if flick_direction:
                speed_type = "FAST" if wrist_movement and wrist_movement >= self._wrist_movement_threshold else "SLOW"
                cv2.putText(frame, f"ONE FINGER - SCROLL UP ({flick_direction}, {speed_type})", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif finger_gesture == "two":
            # Highlight index and middle fingers
            middle = points[12]
            cv2.circle(frame, index, 15, (255, 255, 0), 3)
            cv2.circle(frame, middle, 15, (255, 255, 0), 3)
            if flick_direction:
                speed_type = "FAST" if wrist_movement and wrist_movement >= self._wrist_movement_threshold else "SLOW"
                cv2.putText(frame, f"TWO FINGERS - SCROLL DOWN ({flick_direction}, {speed_type})", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        self._last_state = state_label
        self._last_ratio = pinch_ratio

        return {
            "label": state_label,
            "ratio": pinch_ratio,
            "midpoint": midpoint,
            "inside_box": inside_box,
            "finger_gesture": finger_gesture,
            "flick_direction": flick_direction,
            "wrist_movement": wrist_movement,
        }

    def _map_to_screen(self, midpoint, control_box):
        left, top, right, bottom = control_box
        width = max(1, right - left)
        height = max(1, bottom - top)

        norm_x = np.clip((midpoint[0] - left) / width, 0.0, 1.0)
        norm_y = np.clip((midpoint[1] - top) / height, 0.0, 1.0)

        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        return screen_x, screen_y

    def _move_cursor(self, target):
        target_x, target_y = target
        self.cursor_x = int(self.cursor_x + (target_x - self.cursor_x) * self.smooth_factor)
        self.cursor_y = int(self.cursor_y + (target_y - self.cursor_y) * self.smooth_factor)
        pyautogui.moveTo(self.cursor_x, self.cursor_y)
    
    def _is_finger_extended(self, points, finger_id):
        """
        Check if a finger is extended.
        finger_id: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
        """
        # MediaPipe hand landmarks:
        # Thumb: 4 (tip), 3 (IP), 2 (MP), 1 (CMC)
        # Index: 8 (tip), 7 (PIP), 6 (MCP), 5 (CMC)
        # Middle: 12 (tip), 11 (PIP), 10 (MCP), 9 (CMC)
        # Ring: 16 (tip), 15 (PIP), 14 (MCP), 13 (CMC)
        # Pinky: 20 (tip), 19 (PIP), 18 (MCP), 17 (CMC)
        
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 7, 11, 15, 19]
        finger_mcps = [2, 6, 10, 14, 18]
        
        if finger_id == 0:  # Thumb (special case - check x coordinate relative to wrist)
            tip = points[finger_tips[0]]
            wrist = points[0]  # Wrist landmark
            # For thumb, check if tip is further from wrist than IP joint
            ip = points[finger_pips[0]]
            tip_dist = abs(tip[0] - wrist[0])
            ip_dist = abs(ip[0] - wrist[0])
            return tip_dist > ip_dist
        else:
            tip_idx = finger_tips[finger_id]
            pip_idx = finger_pips[finger_id]
            mcp_idx = finger_mcps[finger_id]
            
            tip = points[tip_idx]
            pip = points[pip_idx]
            mcp = points[mcp_idx]
            
            # Finger is extended if tip is above PIP (y is smaller in image coordinates)
            # Also check that tip is above MCP for more reliable detection
            return tip[1] < pip[1] and tip[1] < mcp[1]
    
    def _detect_finger_gesture(self, points):
        """
        Detect finger counting gesture with occlusion handling.
        Returns: "one" (index only), "two" (index + middle), or None
        """
        index_extended = self._is_finger_extended(points, 1)
        middle_extended = self._is_finger_extended(points, 2)
        thumb_extended = self._is_finger_extended(points, 0)
        ring_extended = self._is_finger_extended(points, 3)
        pinky_extended = self._is_finger_extended(points, 4)
        
        # Get finger tip positions for occlusion detection
        index_tip = points[8]  # Index finger tip
        middle_tip = points[12]  # Middle finger tip
        ring_tip = points[16]  # Ring finger tip
        pinky_tip = points[20]  # Pinky finger tip
        
        # Check if middle finger might be occluded (behind index)
        # If middle finger tip is close to index finger tip horizontally, it might be occluded
        index_middle_horizontal_dist = abs(index_tip[0] - middle_tip[0])
        index_middle_vertical_dist = abs(index_tip[1] - middle_tip[1])
        
        # Get middle finger PIP to check if it's extended
        middle_pip = points[11]  # Middle finger PIP
        
        # If middle finger is close horizontally to index, it might be occluded
        # Check if middle PIP suggests it's extended (tip above PIP) even if tip detection failed
        middle_possibly_extended = middle_extended
        if index_middle_horizontal_dist < 35:  # Close horizontally (might be occluded)
            # If middle tip is above PIP, it's likely extended even if occluded
            if middle_tip[1] < middle_pip[1]:
                middle_possibly_extended = True
            # Also check if middle is in a position that suggests it's extended but behind index
            elif index_middle_vertical_dist < 60:  # Vertically close
                middle_possibly_extended = True
        
        # Check for "two" - index and middle (accounting for occlusion)
        # Both should be extended, or middle might be occluded behind index
        if index_extended and middle_possibly_extended and not ring_extended and not pinky_extended:
            return "two"
        
        # Check for "one" - only index finger extended (thumb can be extended too)
        # Make sure middle is clearly not extended (not just occluded)
        if index_extended and not ring_extended and not pinky_extended:
            # If middle is far from index horizontally, it's definitely not "two"
            if index_middle_horizontal_dist > 45:
                if not middle_extended:
                    return "one"
            # If middle is close but clearly not extended (tip below PIP)
            elif middle_tip[1] >= middle_pip[1]:  # Middle tip is below PIP (not extended)
                return "one"
            # If middle is not detected as extended and not close to index
            elif not middle_extended and index_middle_horizontal_dist > 30:
                return "one"
        
        return None
    
    def _detect_flick(self, current_pos):
        """
        Detect flick gesture based on hand position history.
        Returns: "up", "down", or None
        """
        if len(self._hand_position_history) < 3:
            return None
        
        # Use recent history for flick detection (last 5-7 positions)
        recent_count = min(7, len(self._hand_position_history))
        recent_history = self._hand_position_history[-recent_count:]
        
        if len(recent_history) < 3:
            return None
        
        # Calculate vertical movement (y direction) and time
        # Format: (finger_x, finger_y, wrist_x, wrist_y, timestamp)
        start_entry = recent_history[0]
        end_entry = recent_history[-1]
        start_finger_y = start_entry[1]  # Finger y position
        end_finger_y = end_entry[1]
        start_time = start_entry[4]  # Timestamp
        end_time = end_entry[4]
        
        total_dy = end_finger_y - start_finger_y  # Positive = moving down, Negative = moving up
        total_dt = end_time - start_time
        
        # Check if movement is significant enough
        if abs(total_dy) < self._flick_threshold or total_dt < 0.05:
            return None
        
        # Check for consistent direction (not just noise)
        # Count how many steps are in the same direction
        up_steps = 0
        down_steps = 0
        for i in range(1, len(recent_history)):
            dy_step = recent_history[i][1] - recent_history[i-1][1]  # Finger y movement
            if dy_step < -2:  # Moving up (threshold to ignore small movements)
                up_steps += 1
            elif dy_step > 2:  # Moving down
                down_steps += 1
        
        # Require at least 60% of steps in the same direction
        total_steps = up_steps + down_steps
        if total_steps == 0:
            return None
        
        if up_steps > down_steps and up_steps / total_steps > 0.6:
            return "up"
        elif down_steps > up_steps and down_steps / total_steps > 0.6:
            return "down"
        
        return None
    
    def _perform_scroll(self, gesture, flick_direction, wrist_movement):
        """
        Perform discrete scroll based on gesture and wrist movement.
        gesture: "one" (scroll up) or "two" (scroll down)
        flick_direction: "up" or "down"
        wrist_movement: pixels wrist moved up (positive = moved up, negative = moved down)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self._last_scroll_time < self._scroll_cooldown:
            return
        
        # Determine fast vs slow based on wrist movement
        # Fast flick: whole hand moves up (wrist_movement >= threshold)
        # Slow flick: only finger moves (wrist_movement < threshold)
        if wrist_movement is not None and wrist_movement >= self._wrist_movement_threshold:
            # Fast flick - whole hand moved up - 6 units
            scroll_amount = self._fast_scroll_amount
        else:
            # Slow flick - only finger moved - 2 units
            scroll_amount = self._slow_scroll_amount
        
        # Perform discrete scroll based on gesture
        # Two fingers = scroll DOWN, One finger = scroll UP
        if gesture == "two":
            # Two fingers - scroll down (positive)
            pyautogui.scroll(scroll_amount)
        elif gesture == "one":
            # One finger - scroll up (negative)
            pyautogui.scroll(-scroll_amount)
        
        self._last_scroll_time = current_time

    def _draw_guides(self, frame, control_box, state_info):
        left, top, right, bottom = control_box
        box_color = {
            "no-hand": (80, 80, 80),
            "outside": (0, 165, 255),
            "open": (255, 255, 255),
            "ready": (0, 208, 255),
            "click": (0, 255, 0),
            "scroll-one": (0, 255, 255),
            "scroll-two": (255, 255, 0),
        }.get(state_info["label"], (200, 200, 200))

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Update instruction text based on state
        if state_info.get("finger_gesture") == "one":
            instruction = "ONE finger: Flick to scroll UP"
        elif state_info.get("finger_gesture") == "two":
            instruction = "TWO fingers: Flick to scroll DOWN"
        else:
            instruction = "Move hand inside the box; almost pinch to steer, pinch to click"
        
        cv2.putText(
            frame,
            instruction,
            (max(10, left), max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        status_text = f"State: {state_info['label'].upper()}"
        if state_info["ratio"] is not None:
            status_text += f" | pinch={state_info['ratio']:.2f}"
        if state_info.get("finger_gesture"):
            status_text += f" | gesture={state_info['finger_gesture']}"
        if state_info.get("flick_direction"):
            status_text += f" | flick={state_info['flick_direction']}"
        if state_info.get("wrist_movement") is not None:
            speed_type = "FAST" if state_info['wrist_movement'] >= self._wrist_movement_threshold else "SLOW"
            status_text += f" | wrist={state_info['wrist_movement']:.0f}px ({speed_type})"
        
        cv2.putText(
            frame,
            status_text,
            (max(10, left), bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
        )

        if state_info.get("midpoint"):
            cv2.circle(frame, state_info["midpoint"], 6, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"Screen: {self.screen_width}x{self.screen_height}",
            (10, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
        )
        cv2.putText(
            frame,
            "ESC/Q to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Simple pinch-gesture cursor MVP.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--box-scale",
        type=float,
        default=0.65,
        help="Fraction of the camera frame reserved for the control box.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.35,
        help="Smoothing factor for cursor motion (0.05-1.0).",
    )
    parser.add_argument(
        "--hover-threshold",
        type=float,
        default=0.32,
        help="Normalized pinch ratio that enables cursor placement.",
    )
    parser.add_argument(
        "--click-threshold",
        type=float,
        default=0.1,
        help="Normalized pinch ratio that triggers a click.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.65,
        help="Minimum detection confidence for MediaPipe Hands.",
    )
    parser.add_argument(
        "--trk-conf",
        type=float,
        default=0.55,
        help="Minimum tracking confidence for MediaPipe Hands.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = GestureMouseController(
        box_scale=args.box_scale,
        smooth_factor=args.smooth,
        hover_ratio=args.hover_threshold,
        click_ratio=args.click_threshold,
        detection_confidence=args.det_conf,
        tracking_confidence=args.trk_conf,
    )
    controller.run(camera_index=args.camera)


if __name__ == "__main__":
    main()

