import cv2
import pyttsx3
import threading
import numpy as np
from ultralytics import YOLO

# 1. Initialize Text-to-Speech
engine = pyttsx3.init()


def speak(text):
    # Using a thread-safe approach for pyttsx3 is tricky,
    # but for simple labels, this works:
    engine.say(text)
    engine.runAndWait()


# 2. Load your YOLO model
model = YOLO('my_model/my_model.pt')

cap = cv2.VideoCapture(0)
last_spoken = ""

# CLAHE Filter Object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- [Gamma Table Setup] ---
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- [Step A: Manual Brightness & Contrast] ---
    # alpha = 1.0 to 3.0 (contrast), beta = 0 to 100 (brightness)
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    # --- [Step B: Advanced Filtering] ---
    # 1. Apply Gamma Correction to the adjusted frame
    frame_gamma = cv2.LUT(adjusted_frame, table)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # --- [Step C: Detection] ---
    results = model(processed_frame, conf=0.6)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw on the original 'frame' for visual clarity
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Voice Feedback
            if label_name != last_spoken:
                threading.Thread(target=speak, args=(label_name,), daemon=True).start()
                last_spoken = label_name

    # Display the result
    # Tip: You can show 'processed_frame' instead of 'frame' to see what the AI sees
    cv2.imshow("Road Sign Voice Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()