from collections import deque
import time
import math

class GestureRecognizer:
    def __init__(self, smooth_frames=5, min_hold_ms=120):
        self.hist = deque(maxlen=smooth_frames)
        self.last_committed = None
        self.last_change_t = 0.0
        self.min_hold_ms = min_hold_ms

    @staticmethod
    def dist(p, q):
        (x1, y1), (x2, y2) = p, q
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def is_right_hand(lms):
        # crude heuristic: pinky MCP (17) is to the right of index MCP (5) for a right hand
        return lms[17][1] > lms[5][1]

    def fingers_up(self, lms):
        tipIds = [4, 8, 12, 16, 20]
        fingers = [0] * 5

        is_right = self.is_right_hand(lms)
        # Thumb check (flip direction for left hand)
        if (lms[4][1] > lms[3][1]) if is_right else (lms[4][1] < lms[3][1]):
            fingers[0] = 1

        # Other fingers: tip y < pip y (up)
        for i, tip in enumerate(tipIds[1:], start=1):
            pip = tip - 2
            if lms[tip][2] < lms[pip][2]:
                fingers[i] = 1
        return fingers

    @staticmethod
    def norm_scale(bbox):
        if not bbox:
            return 1.0
        xmin, ymin, xmax, ymax = bbox
        return max(1.0, float(xmax - xmin))  # width

    def name_gesture(self, lmsList, bbox):
        if not lmsList or len(lmsList) <= 20:
            return None

        P = {pid: (x, y) for pid, x, y in lmsList}
        fingers = self.fingers_up(lmsList)
        up_count = sum(fingers)
        scale = self.norm_scale(bbox)

        # distances (normalized)
        pinch_48 = self.dist(P[4], P[8]) / scale
        index_fold = P[8][1] - P[6][1]

        # rules
        if pinch_48 < 0.18:
            return "click"
        if up_count == 5:
            return "open_palm"
        if up_count == 0:
            return "fist"
        if fingers == [0, 1, 0, 0, 0] or (fingers[1] == 1 and index_fold < 0):
            return "point"
        if fingers == [0, 1, 1, 0, 0]:
            return "peace"

        return {1: "one", 2: "two", 3: "three", 4: "four"}.get(up_count, None)

    def update(self, lmsList, bbox):
        g = self.name_gesture(lmsList, bbox)
        self.hist.append(g)

        if not self.hist:
            return None

        # majority vote
        counts = {}
        for x in self.hist:
            counts[x] = counts.get(x, 0) + 1
        candidate = max(counts, key=counts.get)

        now = time.time() * 1000.0

        if candidate != self.last_committed:
            if counts[candidate] >= max(2, len(self.hist) - 1):
                if (now - self.last_change_t) >= self.min_hold_ms:
                    self.last_committed = candidate
                    self.last_change_t = now
                    return candidate
            self.last_change_t = now
            return None

        return self.last_committed
