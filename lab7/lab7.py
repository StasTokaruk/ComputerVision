import cv2
from ultralytics import YOLO
import easyocr
import os
import re
import threading
import queue
import time
import numpy as np

VIDEO_PATH = "../lab7/video.mp4"
MODEL_PATH = "results/train/weights/best.pt"
OUTPUT_DIR = "saved_plates"

PROCESS_WIDTH, PROCESS_HEIGHT = 1280, 720
FRAME_SKIP = 2
CONF_THRESHOLD = 0.5
OCR_THRESHOLD = 0.4
PADDING = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

#ініціалізація моделей
yolo_model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)
ocr_queue = queue.Queue(maxsize=10)
track_history = {}


# фільтрація результатів OCR
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


# функція фонового потоку для обробки черги та розпізнавання тексту
def ocr_worker():
    while True:
        try:
            track_id, plate_img = ocr_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # оптимізація розміру зображення для EasyOCR (приведення висоти до 64px)
        h, w = plate_img.shape[:2]
        if h > 64:
            scale = 64 / h
            plate_img = cv2.resize(plate_img, (int(w * scale), 64))

        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        try:
            ocr_result = reader.readtext(gray_plate, detail=1)
        except Exception:
            ocr_queue.task_done()
            continue

        best_text, best_conf = "", 0.0
        for (_, text, conf) in ocr_result:
            cleaned = clean_text(text)
            if conf > OCR_THRESHOLD and len(cleaned) > 3:
                best_text, best_conf = cleaned, conf
                break

                # оновлення історії, якщо знайдено кращий результат
        if best_text:
            if track_id not in track_history or best_conf > track_history[track_id]['conf']:
                track_history[track_id] = {
                    'text': best_text, 'conf': best_conf,
                    'last_seen': time.time(), 'in_zone': True
                }
                cv2.imwrite(f"{OUTPUT_DIR}/ID_{track_id}_{best_text}.jpg", plate_img)
                print(f"ID {track_id}: {best_text}")

        ocr_queue.task_done()


# запуск фонового потоку OCR
threading.Thread(target=ocr_worker, daemon=True).start()

# головний цикл обробки відео
cap = cv2.VideoCapture(VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = orig_w / PROCESS_WIDTH
scale_y = orig_h / PROCESS_HEIGHT

# налаштування ROI
zone_points = np.array([
    [int(orig_w * 0.20), int(orig_h * 0.60)],
    [int(orig_w * 0.80), int(orig_h * 0.60)],
    [int(orig_w * 0.95), int(orig_h * 0.95)],
    [int(orig_w * 0.05), int(orig_h * 0.95)]
], np.int32)


# перевірка чи точка знаходиться всередині полігону
def is_center_in_zone(cx, cy, zone):
    return cv2.pointPolygonTest(zone, (cx, cy), False) >= 0


window_name = "lab7"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

frame_count = 0
last_boxes, last_track_ids = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # зменшення кадру для прискорення YOLO
    frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    # запуск трекінгу кожен N-й кадр
    if frame_count % FRAME_SKIP == 0:
        results = yolo_model.track(frame_resized, persist=True, verbose=False, conf=CONF_THRESHOLD)
        if results[0].boxes.id is not None:
            last_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            last_track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            last_boxes, last_track_ids = [], []

    # візуалізація зони
    cv2.polylines(frame, [zone_points], isClosed=True, color=(255, 0, 0), thickness=3)

    if len(last_boxes) > 0:
        for box, track_id in zip(last_boxes, last_track_ids):
            # масштабування координат назад до оригінального розміру
            bx1, by1, bx2, by2 = box
            x1 = max(0, int(bx1 * scale_x))
            y1 = max(0, int(by1 * scale_y))
            x2 = min(orig_w, int(bx2 * scale_x))
            y2 = min(orig_h, int(by2 * scale_y))

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_zone = is_center_in_zone(cx, cy, zone_points)

            # логіка відправки на OCR: якщо в зоні та потребує оновлення
            if in_zone:
                should_detect = (track_id not in track_history)
                if not should_detect and track_history[track_id]['conf'] < 0.85:
                    if time.time() - track_history[track_id]['last_seen'] > 1.0:
                        should_detect = True

                if should_detect and not ocr_queue.full():
                    plate_roi = frame[max(0, y1 - PADDING):min(orig_h, y2 + PADDING),
                                max(0, x1 - PADDING):min(orig_w, x2 + PADDING)].copy()

                    if plate_roi.size > 0:
                        ocr_queue.put((track_id, plate_roi))
                        if track_id not in track_history:
                            track_history[track_id] = {'text': "Scanning", 'conf': 0, 'last_seen': time.time()}

            # візуалізація рамок та тексту
            label = ""
            box_color = (0, 255, 255) if in_zone else (100, 100, 100)

            if track_id in track_history:
                if track_history[track_id]['conf'] > 0:
                    label = track_history[track_id]['text']
                    box_color = (0, 255, 0)
                else:
                    label = "Scanning"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            if label:
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
