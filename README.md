# Game-Controller-Using-Hand-Gesture
"Game Controller Using Hand Gestures" is an innovative project developed using Python, OpenCV, and MediaPipe, aiming to create a hands-free method to interact with computer games or applications using natural hand movements.By leveraging real-time hand landmark detection and gesture recognition, this system allows users to control basic game actions (like moving up, down, left, and right) through intuitive finger gestures, without the need for physical controllers or keyboards.

🎯 Key Features
-  Real-Time Hand Tracking: Uses a webcam to continuously track hand movements.
-  Gesture-Based Controls: Detects the position of the index finger to simulate arrow key presses.
-  Cross-Application Integration: Works with any game or application that uses keyboard controls.
-  Human-Computer Interaction: Demonstrates natural interaction between user and system through vision-based input.

🛠️ Tech Stack
-  Python: Core programming language.
-  OpenCV: For real-time image processing and video capture.
-  MediaPipe: For accurate and efficient hand landmark detection.
-  PyAutoGUI: To simulate keyboard inputs based on detected gestures.

📌 Use Cases
-  Accessible gaming interfaces
-  Interactive demos or educational tools
-  Prototypes for gesture-based user interfaces
-  Human-computer interaction experiments
This project uses real-time hand gesture recognition to control game actions using Python, OpenCV, and MediaPipe.

## Features
- Real-time video capture and gesture detection
- Controls: up, down, left, right using index finger movements
- Integration with keyboard control using pyautogui

## How it works
  -The webcam captures video.
  -MediaPipe detects hand landmarks.
  -Index finger tip position is tracked.
  -Specific gesture zones trigger keyboard arrow presses.

## Requirements
-  Python 3.7+
-  OpenCV
-  MediaPipe
-  PyAutoGUI

