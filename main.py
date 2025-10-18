import cv2
import mediapipe as mp
import math
import pyautogui
import numpy as np
import time


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
                    # Draw finger tips with different colors
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id in self.tipIds:
                            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmsList, bbox

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        if not getattr(self, "lmsList", None) or len(self.lmsList) <= max(p1, p2):
            return None, frame, None

        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]


def detect_screen_rectangle(frame, min_area_ratio=0.2):
    """
    Detects the largest rectangle in the frame (assumed to be the monitor/screen).
    
    Args:
        frame: Input frame from camera
        min_area_ratio: Minimum area as ratio of frame size (to filter small rectangles)
    
    Returns:
        corners: List of 4 corner points [(x,y), (x,y), (x,y), (x,y)] ordered as 
                 [top-left, top-right, bottom-right, bottom-left], or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter and find largest rectangular contour
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * min_area_ratio
    
    largest_rect = None
    largest_area = 0
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a quadrilateral (4 points)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            
            # Check if area is large enough and larger than previous largest
            if area > min_area and area > largest_area:
                largest_area = area
                largest_rect = approx
    
    if largest_rect is None:
        return None
    
    # Extract corners and order them: top-left, top-right, bottom-right, bottom-left
    corners = largest_rect.reshape(4, 2)
    corners = order_points(corners)
    
    return corners


def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left
    """
    # Sort by y-coordinate
    pts_sorted = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top_pts = pts_sorted[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
    tl, tr = top_pts
    
    # Bottom two points
    bottom_pts = pts_sorted[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
    bl, br = bottom_pts
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def map_to_screen(pinch_x, pinch_y, screen_corners, screen_width, screen_height):
    """
    Maps camera coordinates to screen coordinates using detected screen corners.
    Uses perspective transformation for accurate mapping.
    """
    # Source points (detected screen corners in camera view)
    src_points = np.float32(screen_corners)
    
    # Destination points (actual screen coordinates)
    dst_points = np.float32([
        [0, 0],
        [screen_width - 1, 0],
        [screen_width - 1, screen_height - 1],
        [0, screen_height - 1]
    ])
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Transform the pinch point
    point = np.array([[[pinch_x, pinch_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)
    
    screen_x = int(transformed[0][0][0])
    screen_y = int(transformed[0][0][1])
    
    # Clamp to screen boundaries
    screen_x = max(0, min(screen_x, screen_width - 1))
    screen_y = max(0, min(screen_y, screen_height - 1))
    
    return screen_x, screen_y


def calibrate_screen_detection(cap, detector):
    """
    Detects the screen/monitor in the camera view.
    Returns the corners of the detected screen.
    """
    print("\n" + "="*60)
    print("SCREEN DETECTION MODE")
    print("="*60)
    print("Instructions:")
    print("1. Position camera so it can see your monitor/screen")
    print("2. Make sure the screen is the largest rectangle visible")
    print("3. Press SPACE when ready to detect")
    print("4. Press 'R' to retry detection")
    print("5. Press ESC to cancel")
    print("="*60 + "\n")
    
    detected_corners = None
    detection_confirmed = False
    
    while not detection_confirmed:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Try to detect screen
        temp_corners = detect_screen_rectangle(frame, min_area_ratio=0.15)
        
        if temp_corners is not None:
            # Draw detected rectangle
            pts = temp_corners.astype(np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)
            
            # Draw corner circles
            for i, corner in enumerate(pts):
                cv2.circle(display_frame, tuple(corner), 10, (0, 255, 255), -1)
                cv2.putText(display_frame, str(i), tuple(corner + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Calculate area percentage
            screen_area = cv2.contourArea(temp_corners)
            frame_area = frame.shape[0] * frame.shape[1]
            area_percent = (screen_area / frame_area) * 100
            
            cv2.putText(display_frame, f"Screen detected! Area: {area_percent:.1f}%", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to confirm", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "No screen detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, "Adjust camera position", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Press 'R' to retry | ESC to cancel", (20, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow('Screen Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None
        elif key == ord(' '):  # SPACE - confirm detection
            if temp_corners is not None:
                detected_corners = temp_corners
                detection_confirmed = True
                print("✓ Screen detected and confirmed!")
        elif key == ord('r') or key == ord('R'):  # Retry
            print("Retrying detection...")
            continue
    
    return detected_corners


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = HandTrackingDynamic(maxHands=1)
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {cam_width}x{cam_height}")
    
    # Initialize screen corners - will be detected continuously
    screen_corners = None
    
    # Gesture state tracking
    pinch_state = "none"  # none, start, hold, release
    pinch_start_time = 0
    pinch_hold_threshold = 0.3  # seconds to hold before drag mode
    click_threshold = 0.2  # seconds for quick click
    last_pinch_pos = None
    drag_started = False
    click_performed = False
    
    # Main tracking loop
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    pyautogui.FAILSAFE = False
    show_window = 1
    
    print("\n" + "="*60)
    print("TRACKING MODE")
    print("Cursor follows midpoint between thumb and index finger")
    print("Quick pinch (< 0.2s) = CLICK")
    print("Hold pinch + move (> 0.3s) = DRAG mode")
    print("Screen detection: CONTINUOUS (automatically finds screen)")
    print("Press 'W' to toggle window | ESC to exit")
    print("Pinch sensitivity: Adjust by changing pinch_threshold in code (default: 0.15)")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_height, frame_width = frame.shape[:2]
        
        # Continuously detect screen rectangle
        temp_corners = detect_screen_rectangle(frame, min_area_ratio=0.10)
        if temp_corners is not None:
            # If we don't have previous corners, use the new ones
            if screen_corners is None:
                screen_corners = temp_corners
            else:
                # Calculate center distance between old and new corners
                old_center = np.mean(screen_corners, axis=0)
                new_center = np.mean(temp_corners, axis=0)
                center_distance = np.linalg.norm(new_center - old_center)
                
                # Only update if the center hasn't moved too much (prevents jitter)
                # Use a more lenient threshold for continuous detection
                if center_distance < 150:  # Increased threshold for better tracking
                    screen_corners = temp_corners
                # If the new detection is significantly different, it might be a better detection
                elif center_distance > 200:  # Large movement - might be camera repositioning
                    screen_corners = temp_corners
        
        # Process hand tracking
        frame = detector.findFingers(frame, draw=show_window)
        lmsList, bbox = detector.findPosition(frame, draw=show_window)
        
        # Draw detected screen boundary
        if show_window and screen_corners is not None:
            pts = screen_corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            for corner in pts:
                cv2.circle(frame, tuple(corner), 6, (0, 255, 0), -1)
            
            # Add "Screen Detected" text near the rectangle
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))
            cv2.putText(frame, "SCREEN DETECTED", (center_x - 80, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif show_window:
            # Show "Looking for screen..." message
            cv2.putText(frame, "Looking for screen...", (frame_width//2 - 100, frame_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Detect hand position and pinch state
        current_time = time.time()
        hand_detected = False
        hand_x, hand_y = 0, 0
        pinch_detected = False
        pinch_strength = 0
        
        if lmsList and len(lmsList) > 8:
            # Get thumb and index finger positions
            thumb_x, thumb_y = lmsList[4][1], lmsList[4][2]  # Thumb tip
            index_x, index_y = lmsList[8][1], lmsList[8][2]  # Index finger tip
            
            # Calculate midpoint between thumb and index finger for cursor positioning
            hand_x = (thumb_x + index_x) // 2
            hand_y = (thumb_y + index_y) // 2
            hand_detected = True
            
            # Get distance between thumb and index finger for pinch detection
            length, frame, info = detector.findDistance(4, 8, frame, draw=True)
            
            if length is not None and bbox:
                hand_w = max(1, bbox[2] - bbox[0])
                hand_h = max(1, bbox[3] - bbox[1])
                hand_size = max(hand_w, hand_h)
                
                # More robust pinch detection using multiple criteria
                pinch_ratio = length / hand_size
                pinch_threshold = 0.15  # Adjust this value for sensitivity
                
                if pinch_ratio < pinch_threshold:
                    pinch_detected = True
                    pinch_strength = 1.0 - (pinch_ratio / pinch_threshold)  # 0-1 strength
                    
                    # Handle gesture state transitions
                    if pinch_state == "none":
                        pinch_state = "start"
                        pinch_start_time = current_time
                        click_performed = False
                        drag_started = False
                    elif pinch_state == "start":
                        # Check if we should transition to hold state
                        if current_time - pinch_start_time > pinch_hold_threshold:
                            pinch_state = "hold"
                            if not drag_started:
                                drag_started = True
                                pyautogui.mouseDown()
                                print("Drag started")
                    elif pinch_state == "hold":
                        # Continue dragging
                        pass
                else:
                    # Pinch released
                    if pinch_state == "start":
                        # Quick pinch - perform click
                        if current_time - pinch_start_time < click_threshold and not click_performed:
                            pyautogui.click()
                            click_performed = True
                            print("Click performed")
                    elif pinch_state == "hold":
                        # End drag
                        pyautogui.mouseUp()
                        print("Drag ended")
                    
                    pinch_state = "none"
                    drag_started = False
        
        # Always move cursor when hand is detected and screen is found
        if hand_detected and screen_corners is not None:
            # Map hand position to screen coordinates
            screen_x, screen_y = map_to_screen(hand_x, hand_y, screen_corners,
                                              screen_width, screen_height)
            
            # Move cursor with smoothing for better tracking
            current_x, current_y = pyautogui.position()
            # Add some smoothing to cursor movement
            smooth_factor = 0.3
            new_x = int(current_x + (screen_x - current_x) * smooth_factor)
            new_y = int(current_y + (screen_y - current_y) * smooth_factor)
            pyautogui.moveTo(new_x, new_y)
            
            if show_window:
                # Draw thumb and index finger positions
                cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)  # Blue for thumb
                cv2.circle(frame, (index_x, index_y), 8, (0, 0, 255), -1)  # Red for index
                
                # Draw midpoint (cursor position)
                cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (hand_x, hand_y), 15, (255, 255, 255), 2)
                
                # Draw line between thumb and index finger
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 255), 2)
                
                # Draw pinch indicator if pinching
                if pinch_detected:
                    if pinch_state == "start":
                        color = (0, 255, 255)  # Yellow for start
                        cv2.circle(frame, (hand_x, hand_y), 20, color, 3)
                    elif pinch_state == "hold":
                        color = (0, 0, 255)  # Red for drag
                        cv2.circle(frame, (hand_x, hand_y), 25, color, 4)
                
                # Display information based on state
                if pinch_detected:
                    if pinch_state == "start":
                        hold_time = current_time - pinch_start_time
                        cv2.putText(frame, f"PINCH - Hold for drag ({hold_time:.1f}s)", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif pinch_state == "hold":
                        cv2.putText(frame, "DRAGGING MODE", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"PINCH DETECTED! Strength: {pinch_strength:.2f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "HAND DETECTED - Move to control cursor", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Midpoint: ({hand_x}, {hand_y})", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if length is not None:
                    cv2.putText(frame, f"Distance: {length:.1f}px", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw line from hand to mapped screen location
                screen_cam_x = int((screen_x / screen_width) * (screen_corners[2][0] - screen_corners[0][0]) + screen_corners[0][0])
                screen_cam_y = int((screen_y / screen_height) * (screen_corners[2][1] - screen_corners[0][1]) + screen_corners[0][1])
                cv2.line(frame, (hand_x, hand_y), (screen_cam_x, screen_cam_y), (255, 0, 255), 3)
                
                # Draw crosshair at target location
                cv2.line(frame, (screen_cam_x-10, screen_cam_y), (screen_cam_x+10, screen_cam_y), (0, 255, 255), 2)
                cv2.line(frame, (screen_cam_x, screen_cam_y-10), (screen_cam_x, screen_cam_y+10), (0, 255, 255), 2)
            
            print(f"Midpoint at ({hand_x}, {hand_y}) -> Screen ({screen_x}, {screen_y}) | Pinch: {pinch_detected} | State: {pinch_state} | Distance: {length:.1f}px")
        else:
            if show_window and lmsList:
                # Show hand is detected but screen not found
                cv2.putText(frame, "Hand detected - Waiting for screen detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if show_window:
            # Status indicator
            if screen_corners is not None:
                status_color = (0, 255, 0)
                status_text = "READY - Screen detected"
            else:
                status_color = (0, 255, 255)  # Yellow for detecting
                status_text = "DETECTING SCREEN..."
            
            cv2.putText(frame, f"Status: {status_text}", (10, frame_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "Press 'W' to toggle window | ESC to exit", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Hand Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('w') or key == ord('W'):  # Toggle window
            show_window = not show_window
            if not show_window:
                cv2.destroyWindow('Hand Tracking')
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()