import cv2
import pygame
import threading
import time
import os
import numpy as np
from ultralytics import YOLO

# ၁။ Sound System
pygame.mixer.init()

def play_sound(label_name):
    sound_path = f"sounds/{label_name}.mp3"
    if os.path.exists(sound_path):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()


# ၂။ Distance Calculation Parameters
# (မှတ်ချက် - ဤတန်ဖိုးများသည် ကင်မရာအမျိုးအစားပေါ်မူတည်၍ အနည်းငယ်ကွာခြားနိုင်သည်)
KNOWN_WIDTH = 0.6  # ဆိုင်းဘုတ်တစ်ခု၏ ပျမ်းမျှအကျယ် (၀.၆ မီတာ သတ်မှတ်ချက်)
FOCAL_LENGTH = 800  # ကင်မရာ၏ Focal Length (Calibration မရှိပါက ၈၀၀ ခန့်ဖြင့် စမ်းသပ်နိုင်သည်)


def estimate_distance(pixel_width):
    # Distance = (Real Width * Focal Length) / Pixel Width
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width


# ၃။ Model Load
model = YOLO("my_model/my_model.pt")

# ၄။ Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Preprocessing Parameters
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

last_spoken = ""
last_sound_time = 0
cooldown = 3  # အသံခဏခဏမထွက်စေရန် စက္ကန့်တိုးထားသည်

while cap.isOpened():


    start_time = time.time()
    success, frame = cap.read()
    if not success: break

    # Preprocessing
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    frame_gamma = cv2.LUT(blurred, table)
    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # YOLO Tracking
    results = model.track(processed_frame, persist=True, conf=0.25, imgsz=640, verbose=False)

    detected = False

    for r in results:
        if r.boxes is None: continue

        for box in r.boxes:
            detected = True
            class_id = int(box.cls[0])
            label_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- [Distance Calculation Addition] ---
            box_width = x2 - x1
            distance = estimate_distance(box_width)

            # Distance အလိုက် အရောင်ပြောင်းလဲခြင်း
            color = (0, 255, 0)  # အဝေး (အစိမ်း)
            if distance < 10: color = (0, 165, 255)  # အလတ် (လိမ္မော်)
            if distance < 5: color = (0, 0, 255)  # အနီး (အနီ)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} {distance:.1f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Sound Logic: အကွာအဝေး ၁၅ မီတာအတွင်း ရောက်မှသာ သတိပေးမည်
            current_time = time.time()
            if distance < 15:
                time_passed = current_time - last_sound_time
                if label_name != last_spoken or time_passed > cooldown:
                    threading.Thread(target=play_sound, args=(label_name,)).start()
                    last_spoken = label_name
                    last_sound_time = current_time

    if not detected:
        last_spoken = ""

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Distance Aware Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()