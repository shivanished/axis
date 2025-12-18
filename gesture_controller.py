"""
Core gesture control logic - separated from UI concerns.
"""
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from OneEuroFilter import OneEuroFilter


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
        use_one_euro=True,
        min_cutoff=1.0,
        beta=0.007,
        use_interpolation=False,
        buffer_size=5,
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
        self.use_one_euro = use_one_euro
        self.use_interpolation = use_interpolation

        self.detector = SimpleHandTracker(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )

        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        start_x, start_y = pyautogui.position()
        self.cursor_x = start_x
        self.cursor_y = start_y

        # Initialize One Euro Filters for X and Y coordinates
        # freq=30 assumes ~30fps processing rate
        if self.use_one_euro:
            self.filter_x = OneEuroFilter(freq=30, mincutoff=min_cutoff, beta=beta, dcutoff=1.0)
            self.filter_y = OneEuroFilter(freq=30, mincutoff=min_cutoff, beta=beta, dcutoff=1.0)

        self._click_latched = False
        self._drag_active = False
        self._drag_start_pos = None
        self._pinch_start_pos = None
        self._last_state = "no-hand"
        self._last_ratio = None
        
        # Scrolling state
        self._hand_position_history = []
        self._max_history = 15
        self._flick_threshold = 30
        self._last_scroll_time = time.time()
        self._scroll_cooldown = 0.2
        self._wrist_movement_threshold = 15
        self._slow_scroll_amount = 2
        self._fast_scroll_amount = 6

        # Frame buffering for smooth interpolation
        self._position_buffer = []  # Buffer of (x, y, timestamp) tuples
        self._buffer_size = buffer_size  # Number of frames to buffer (adds latency but increases smoothness)
        self._last_interpolation_time = time.time()

    def update_smoothing_params(self, use_one_euro=None, min_cutoff=None, beta=None, use_interpolation=None):
        """
        Update smoothing parameters on the fly without restarting the controller.

        Args:
            use_one_euro: Whether to use One Euro Filter (if None, keep current)
            min_cutoff: Minimum cutoff frequency for One Euro Filter (if None, keep current)
            beta: Beta parameter for One Euro Filter (if None, keep current)
            use_interpolation: Whether to use frame interpolation (if None, keep current)
        """
        if use_one_euro is not None:
            # If switching filter mode, reinitialize filters
            if use_one_euro != self.use_one_euro:
                self.use_one_euro = use_one_euro
                if use_one_euro:
                    # Initialize new filters with current position
                    self.filter_x = OneEuroFilter(
                        freq=30,
                        mincutoff=min_cutoff if min_cutoff is not None else 1.0,
                        beta=beta if beta is not None else 0.007,
                        dcutoff=1.0
                    )
                    self.filter_y = OneEuroFilter(
                        freq=30,
                        mincutoff=min_cutoff if min_cutoff is not None else 1.0,
                        beta=beta if beta is not None else 0.007,
                        dcutoff=1.0
                    )

        # Update existing filter parameters using official library's setter methods
        if self.use_one_euro and (min_cutoff is not None or beta is not None):
            if min_cutoff is not None:
                self.filter_x.setMinCutoff(min_cutoff)
                self.filter_y.setMinCutoff(min_cutoff)
            if beta is not None:
                self.filter_x.setBeta(beta)
                self.filter_y.setBeta(beta)

        # Update interpolation setting
        if use_interpolation is not None:
            self.use_interpolation = use_interpolation
            # Clear buffer when toggling interpolation to avoid stale data
            if use_interpolation:
                self._position_buffer = []

    def process_frame(self, frame, mirror=True):
        """
        Process a single frame and return the annotated frame and state info.
        This is the core processing logic separated from UI concerns.
        """
        if mirror:
            frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]
        control_box = self._control_box(frame_width, frame_height)

        hand_info = self.detector.locate(frame, draw=True)
        if hand_info:
            state_info = self._handle_hand(frame, hand_info, control_box)
        else:
            self._click_latched = False
            self._hand_position_history = []
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
        return frame, state_info

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

        finger_gesture = self._detect_finger_gesture(points)
        wrist = points[0]
        
        current_time = time.time()
        self._hand_position_history.append((midpoint[0], midpoint[1], wrist[0], wrist[1], current_time))
        if len(self._hand_position_history) > self._max_history:
            self._hand_position_history.pop(0)
        
        flick_direction = self._detect_flick(midpoint)
        
        wrist_movement = None
        if finger_gesture and flick_direction and len(self._hand_position_history) >= 2:
            window_size = min(5, len(self._hand_position_history))
            recent_window = self._hand_position_history[-window_size:]
            if len(recent_window) >= 2:
                start_entry = recent_window[0]
                end_entry = recent_window[-1]
                wrist_dy = start_entry[3] - end_entry[3]
                wrist_movement = wrist_dy
        
        if finger_gesture and flick_direction:
            self._perform_scroll(finger_gesture, flick_direction, wrist_movement)
            current_time = time.time()
            self._hand_position_history = [(midpoint[0], midpoint[1], wrist[0], wrist[1], current_time)]

        left, top, right, bottom = control_box
        inside_box = left <= midpoint[0] <= right and top <= midpoint[1] <= bottom

        if not inside_box:
            state_label = "outside"
            self._click_latched = False
            screen_target = None
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            self._drag_start_pos = None
            self._pinch_start_pos = None
        else:
            if pinch_ratio < self.click_ratio:
                state_label = "click"
            elif pinch_ratio < self.hover_ratio:
                state_label = "ready"
            elif finger_gesture:
                state_label = f"scroll-{finger_gesture}"
                screen_target = None
                self._click_latched = False
            else:
                state_label = "open"

            if state_label in ("ready", "click"):
                screen_target = self._map_to_screen(midpoint, control_box)
                self._move_cursor(screen_target)
                
                if state_label == "click":
                    if not self._click_latched:
                        pyautogui.click()
                        self._click_latched = True
                        self._pinch_start_pos = screen_target
                        self._drag_start_pos = screen_target
                    else:
                        if self._pinch_start_pos is not None:
                            move_distance = math.hypot(
                                screen_target[0] - self._pinch_start_pos[0],
                                screen_target[1] - self._pinch_start_pos[1]
                            )
                            if move_distance > 10:
                                if not self._drag_active:
                                    self._drag_active = True
                                    pyautogui.mouseDown()
                                    self._drag_start_pos = screen_target
                    
                    if self._drag_active:
                        pass
                else:
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    self._drag_start_pos = None
                    self._pinch_start_pos = None
            else:
                screen_target = None
                if state_label not in ("scroll-one", "scroll-two"):
                    self._click_latched = False
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    self._drag_start_pos = None
                    self._pinch_start_pos = None

            if state_label != "click":
                if state_label not in ("scroll-one", "scroll-two"):
                    if self._drag_active:
                        pyautogui.mouseUp()
                        self._drag_active = False
                    self._click_latched = False
                    self._pinch_start_pos = None

        cv2.circle(frame, thumb, 8, (0, 165, 255), -1)
        cv2.circle(frame, index, 8, (255, 90, 120), -1)
        cv2.line(frame, thumb, index, (255, 255, 255), 2)
        
        if self._drag_active:
            cv2.circle(frame, midpoint, 15, (0, 0, 255), 3)
            cv2.putText(frame, "DRAGGING", (midpoint[0] - 40, midpoint[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(frame, midpoint, 10, (0, 255, 0), 2)
        
        if finger_gesture == "one":
            cv2.circle(frame, index, 15, (0, 255, 255), 3)
            if flick_direction:
                speed_type = "FAST" if wrist_movement and wrist_movement >= self._wrist_movement_threshold else "SLOW"
                cv2.putText(frame, f"ONE FINGER - SCROLL UP ({flick_direction}, {speed_type})", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif finger_gesture == "two":
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
        """
        Move cursor with optional frame buffering and interpolation.
        If interpolation is disabled, uses standard One Euro Filter smoothing.
        """
        target_x, target_y = target
        current_time = time.time()

        # Standard mode (no interpolation) - default behavior
        if not self.use_interpolation:
            if self.use_one_euro:
                # Use One Euro Filter for adaptive smoothing
                smoothed_x = self.filter_x(target_x, current_time)
                smoothed_y = self.filter_y(target_y, current_time)
                self.cursor_x = int(smoothed_x)
                self.cursor_y = int(smoothed_y)
            else:
                # Fall back to simple exponential smoothing
                self.cursor_x = int(self.cursor_x + (target_x - self.cursor_x) * self.smooth_factor)
                self.cursor_y = int(self.cursor_y + (target_y - self.cursor_y) * self.smooth_factor)

            pyautogui.moveTo(self.cursor_x, self.cursor_y)
            return

        # Interpolation mode (optional, adds latency but smoother)
        # Add target to buffer
        self._position_buffer.append((target_x, target_y, current_time))

        # Keep buffer at specified size
        if len(self._position_buffer) > self._buffer_size:
            self._position_buffer.pop(0)

        # Need at least 2 positions to interpolate
        if len(self._position_buffer) < 2:
            # Not enough frames yet, just use current position
            if self.use_one_euro:
                smoothed_x = self.filter_x(target_x, current_time)
                smoothed_y = self.filter_y(target_y, current_time)
                self.cursor_x = smoothed_x
                self.cursor_y = smoothed_y
            else:
                self.cursor_x = target_x
                self.cursor_y = target_y
            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))
            return

        # Interpolate between buffered positions
        # Use the oldest two positions in the buffer for smooth interpolation
        pos0_x, pos0_y, time0 = self._position_buffer[0]
        pos1_x, pos1_y, time1 = self._position_buffer[1]

        # Calculate interpolation factor based on time
        time_delta = time1 - time0
        if time_delta > 0:
            # We want to smoothly transition from pos0 to pos1
            elapsed = current_time - time0
            t = min(1.0, elapsed / time_delta)  # Clamp to [0, 1]
        else:
            t = 1.0

        # Linear interpolation with sub-pixel precision
        interp_x = pos0_x + (pos1_x - pos0_x) * t
        interp_y = pos0_y + (pos1_y - pos0_y) * t

        # Apply One Euro Filter to the interpolated position for extra smoothness
        if self.use_one_euro:
            smoothed_x = self.filter_x(interp_x, current_time)
            smoothed_y = self.filter_y(interp_y, current_time)
            self.cursor_x = smoothed_x
            self.cursor_y = smoothed_y
        else:
            # Simple exponential smoothing on interpolated position
            self.cursor_x = self.cursor_x + (interp_x - self.cursor_x) * self.smooth_factor
            self.cursor_y = self.cursor_y + (interp_y - self.cursor_y) * self.smooth_factor

        # Move cursor (only convert to int at the final step)
        pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))
    
    def _is_finger_extended(self, points, finger_id):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 7, 11, 15, 19]
        finger_mcps = [2, 6, 10, 14, 18]
        
        if finger_id == 0:
            tip = points[finger_tips[0]]
            wrist = points[0]
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
            
            return tip[1] < pip[1] and tip[1] < mcp[1]
    
    def _detect_finger_gesture(self, points):
        index_extended = self._is_finger_extended(points, 1)
        middle_extended = self._is_finger_extended(points, 2)
        ring_extended = self._is_finger_extended(points, 3)
        pinky_extended = self._is_finger_extended(points, 4)
        
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        
        index_middle_horizontal_dist = abs(index_tip[0] - middle_tip[0])
        index_middle_vertical_dist = abs(index_tip[1] - middle_tip[1])
        
        middle_pip = points[11]
        
        middle_possibly_extended = middle_extended
        if index_middle_horizontal_dist < 35:
            if middle_tip[1] < middle_pip[1]:
                middle_possibly_extended = True
            elif index_middle_vertical_dist < 60:
                middle_possibly_extended = True
        
        if index_extended and middle_possibly_extended and not ring_extended and not pinky_extended:
            return "two"
        
        if index_extended and not ring_extended and not pinky_extended:
            if index_middle_horizontal_dist > 45:
                if not middle_extended:
                    return "one"
            elif middle_tip[1] >= middle_pip[1]:
                return "one"
            elif not middle_extended and index_middle_horizontal_dist > 30:
                return "one"
        
        return None
    
    def _detect_flick(self, current_pos):
        if len(self._hand_position_history) < 3:
            return None
        
        recent_count = min(7, len(self._hand_position_history))
        recent_history = self._hand_position_history[-recent_count:]
        
        if len(recent_history) < 3:
            return None
        
        start_entry = recent_history[0]
        end_entry = recent_history[-1]
        start_finger_y = start_entry[1]
        end_finger_y = end_entry[1]
        start_time = start_entry[4]
        end_time = end_entry[4]
        
        total_dy = end_finger_y - start_finger_y
        total_dt = end_time - start_time
        
        if abs(total_dy) < self._flick_threshold or total_dt < 0.05:
            return None
        
        up_steps = 0
        down_steps = 0
        for i in range(1, len(recent_history)):
            dy_step = recent_history[i][1] - recent_history[i-1][1]
            if dy_step < -2:
                up_steps += 1
            elif dy_step > 2:
                down_steps += 1
        
        total_steps = up_steps + down_steps
        if total_steps == 0:
            return None
        
        if up_steps > down_steps and up_steps / total_steps > 0.6:
            return "up"
        elif down_steps > up_steps and down_steps / total_steps > 0.6:
            return "down"
        
        return None
    
    def _perform_scroll(self, gesture, flick_direction, wrist_movement):
        current_time = time.time()
        
        if current_time - self._last_scroll_time < self._scroll_cooldown:
            return
        
        if wrist_movement is not None and wrist_movement >= self._wrist_movement_threshold:
            scroll_amount = self._fast_scroll_amount
        else:
            scroll_amount = self._slow_scroll_amount
        
        if gesture == "two":
            pyautogui.scroll(scroll_amount)
        elif gesture == "one":
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

