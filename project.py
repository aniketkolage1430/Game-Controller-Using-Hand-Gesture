import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe and webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            index_finger_tip = landmarks[8]  # Index finger tip
            x = int(index_finger_tip.x * image.shape[1])
            y = int(index_finger_tip.y * image.shape[0])

            cv2.circle(image, (x, y), 10, (0, 255, 255), cv2.FILLED)

            if y < 150:
                pyautogui.press("up")
                cv2.putText(image, "UP", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif y > 300:
                pyautogui.press("down")
                cv2.putText(image, "DOWN", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif x < 200:
                pyautogui.press("left")
                cv2.putText(image, "LEFT", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif x > 400:
                pyautogui.press("right")
                cv2.putText(image, "RIGHT", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Game Controller", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
