import cv2
import sys
import time

TRACKERS = {
    "CSRT": cv2.TrackerCSRT_create,
    "KCF": cv2.TrackerKCF_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create

}

# current_tracker_name = "CSRT"
# current_tracker_name = "KCF"
current_tracker_name = "MOSSE"

try:
    tracker = TRACKERS[current_tracker_name]()
    print(f"Використовується трекер: {current_tracker_name}")
except AttributeError as e:
    print(f"Помилка трекера: {e}")
    sys.exit()

bbox = None

video_source = 0
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Помилка: Не вдалося відкрити відеопотік.")
    sys.exit()

# Ініціалізація змінних для вимірювання FPS
fps_start_time = time.time()
fps_frame_count = 0

while True:
    # Зчитування поточного кадру
    ret, frame = cap.read()
    if not ret:
        print("Кінець відеопотоку.")
        break

    #Зменшив розмір для покращення продуктивності
    frame = cv2.resize(frame, (640, 480))

    # Очікування вибору ROI
    if bbox is None:
        # За допомогою функції selectROI Обираєм об'єкт
        cv2.putText(frame, "Select ROI and press ENTER", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        bbox = cv2.selectROI(current_tracker_name, frame, False, False)

        # Перевірка вибраної області
        if bbox != (0, 0, 0, 0):
            try:
                tracker.init(frame, bbox)
                print(f"Трекер {current_tracker_name} ініціалізовано.")
            except Exception as e:
                print(f"Помилка ініціалізації трекера: {e}")
                bbox = None
        else:
            print("Вибір ROI скасовано. Завершення роботи.")
            break

    else:
        # Вимірюємо час, витрачений на оновлення
        timer = cv2.getTickCount()
        success, bbox_new = tracker.update(frame)
        fps_frame_count += 1

        # Розрахунок часу обробки
        fps_time_delta = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
        current_fps = 1.0 / fps_time_delta

        # Відображення результатів
        if success:
            (x, y, w, h) = [int(v) for v in bbox_new]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure (Lost)", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Відображення інформації
        cv2.putText(frame, f"{current_tracker_name} Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS: {int(current_fps)}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Відображення кадру
    cv2.imshow(current_tracker_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

total_time = time.time() - fps_start_time
if fps_frame_count > 0 and total_time > 0:
    average_fps = fps_frame_count / total_time
    print("Аналіз результатів:")
    print(f"Метод: {current_tracker_name}")
    print(f"Обробка {fps_frame_count} кадрів.")
    print(f"Загальний час виконання: {total_time:.2f} сек.")
    print(f"Середній FPS: {average_fps:.2f}")