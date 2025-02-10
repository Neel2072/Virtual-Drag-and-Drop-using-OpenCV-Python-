import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Rectangle class
class DragRect:
    def __init__(self, pos, size=200, number=1):
        self.pos = list(pos)  # Center position of the rectangle
        self.size = size      # Size of the rectangle
        self.number = number  # Number to display on the rectangle
        self.dragging = False  # To track if the rectangle is being dragged

    def update(self, cursor):
        self.pos[0], self.pos[1] = cursor[0], cursor[1]  # Update position

    def is_over(self, cursor):
        # Check if the cursor is over the rectangle
        return (self.pos[0] - self.size // 2 < cursor[0] < self.pos[0] + self.size // 2) and \
               (self.pos[1] - self.size // 2 < cursor[1] < self.pos[1] + self.size // 2)

# Create rectangles
rects = [DragRect((i * 250 + 150, 150), number=i + 1) for i in range(5)]
active_rect = None

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']  # Landmark points of the first hand
        if lmList:
            # Calculate distance between index finger tip (8) and middle finger tip (12)
            distance_info = detector.findDistance(lmList[8][:2], lmList[12][:2])
            if isinstance(distance_info, tuple):
                distance = distance_info[0]  # Extract distance from the tuple
            else:
                distance = distance_info  # If only distance is returned

            if distance < 40:  # If fingers are close, drag the rectangle
                cursor = lmList[8]  # Index finger tip (landmark 8)
                if active_rect is None:
                    for rect in rects:
                        if rect.is_over(cursor[:2]):
                            active_rect = rect
                            active_rect.dragging = True
                            break
                if active_rect is not None:
                    active_rect.update(cursor[:2])
            else:
                if active_rect is not None:
                    active_rect.dragging = False
                    active_rect = None
    else:
        if active_rect is not None:
            active_rect.dragging = False
            active_rect = None

    # Draw rectangles
    for rect in rects:
        cx, cy = rect.pos
        size = rect.size
        overlay = img.copy()
        cv2.rectangle(overlay, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (255, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.putText(img, str(rect.number), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()