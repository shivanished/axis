import argparse
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


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
        click_ratio=0.18,
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
        self._last_state = "no-hand"
        self._last_ratio = None

    def run(self, camera_index=0, mirror=True):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to access camera index {camera_index}")

        window_name = "Met Glasses MVP"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 70)
        print("GESTURE CONTROL MVP")
        print("=" * 70)
        print("• Keep your hand inside the on-screen bounding box.")
        print("• Hover the cursor by almost pinching thumb and index (leave small gap).")
        print("• Complete the pinch to trigger a click.")
        print("• Press ESC or Q to exit.")
        print("=" * 70 + "\n")

        while True:
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
                state_info = {
                    "label": "no-hand",
                    "ratio": None,
                    "midpoint": None,
                    "inside_box": False,
                }

            self._draw_guides(frame, control_box, state_info)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

        cap.release()
        cv2.destroyAllWindows()

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

        left, top, right, bottom = control_box
        inside_box = left <= midpoint[0] <= right and top <= midpoint[1] <= bottom

        if not inside_box:
            state_label = "outside"
            self._click_latched = False
            screen_target = None
        else:
            if pinch_ratio < self.click_ratio:
                state_label = "click"
            elif pinch_ratio < self.hover_ratio:
                state_label = "ready"
            else:
                state_label = "open"

            if state_label in ("ready", "click"):
                screen_target = self._map_to_screen(midpoint, control_box)
                self._move_cursor(screen_target)
            else:
                screen_target = None
                self._click_latched = False

            if state_label == "click" and not self._click_latched:
                pyautogui.click()
                self._click_latched = True
            elif state_label != "click":
                self._click_latched = False

        # Visual cues for thumb/index/midpoint
        cv2.circle(frame, thumb, 8, (0, 165, 255), -1)
        cv2.circle(frame, index, 8, (255, 90, 120), -1)
        cv2.line(frame, thumb, index, (255, 255, 255), 2)
        cv2.circle(frame, midpoint, 10, (0, 255, 0), 2)

        self._last_state = state_label
        self._last_ratio = pinch_ratio

        return {
            "label": state_label,
            "ratio": pinch_ratio,
            "midpoint": midpoint,
            "inside_box": inside_box,
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

    def _draw_guides(self, frame, control_box, state_info):
        left, top, right, bottom = control_box
        box_color = {
            "no-hand": (80, 80, 80),
            "outside": (0, 165, 255),
            "open": (255, 255, 255),
            "ready": (0, 208, 255),
            "click": (0, 255, 0),
        }.get(state_info["label"], (200, 200, 200))

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        cv2.putText(
            frame,
            "Move hand inside the box; almost pinch to steer, pinch to click",
            (max(10, left), max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        status_text = f"State: {state_info['label'].upper()}"
        if state_info["ratio"] is not None:
            status_text += f" | pinch={state_info['ratio']:.2f}"
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
        default=0.18,
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

