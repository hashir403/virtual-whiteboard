import cv2
import numpy as np
import mediapipe as mp
import time

# Mediapipe Hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Canvas for drawing
canvas = None
prev_x, prev_y = 0, 0
mode_text = "Idle"

# Brush settings
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
color_index = 0
brush_sizes = [5, 10, 20]
brush_index = 0

def fingers_up(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # [Thumb, Index, Middle, Ring, Pinky]
    landmarks = hand_landmarks.landmark

    # Thumb (x axis check)
    fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x else 0)

    # Other 4 fingers (y axis check)
    for id in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y else 0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            fingers = fingers_up(hand_landmarks)

            x, y = int(lm[8].x * w), int(lm[8].y * h)  # Index finger tip

            # Writing mode (Index finger only)
            if fingers[1] == 1 and sum(fingers) == 1:
                mode_text = "Writing"
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         colors[color_index], brush_sizes[brush_index])
                prev_x, prev_y = x, y

            # Move mode (Index + Middle finger)
            elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                mode_text = "Moving"
                prev_x, prev_y = 0, 0

            # Erase mode (All fingers up)
            elif sum(fingers) == 5:
                mode_text = "Erasing"
                cv2.circle(canvas, (x, y), 50, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            # Idle mode (Fist / no fingers)
            elif sum(fingers) == 0:
                mode_text = "Idle"
                prev_x, prev_y = 0, 0

            else:
                mode_text = "Idle"
                prev_x, prev_y = 0, 0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine drawing with webcam feed
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(bg, fg)

    # Show mode + brush settings
    cv2.putText(frame, f"Mode: {mode_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Color: {colors[color_index]}  Brush: {brush_sizes[brush_index]}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Air Writing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC → exit
        break
    elif key == ord('c'):  # C → change color
        color_index = (color_index + 1) % len(colors)
    elif key == ord('b'):  # B → change brush size
        brush_index = (brush_index + 1) % len(brush_sizes)
    elif key == ord('r'):  # R → reset canvas
        canvas = np.zeros_like(frame)
    elif key == ord('s'):  # S → save current drawing
        timestamp = int(time.time())
        filename = f"drawing_{timestamp}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Drawing saved as {filename}")

cap.release()
cv2.destroyAllWindows()
