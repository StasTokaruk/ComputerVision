import cv2
import numpy as np
import imutils
import time
import argparse

class PersonDetector:
    def __init__(self, prototxt_path, model_path, conf_threshold=0.4):
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.conf_threshold = conf_threshold
        self.input_size = (400, 400)

    def detect_people(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, self.input_size),0.007843, self.input_size, 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        person_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    person_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = f"Person: {confidence * 100:.1f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY),self.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
        return frame, person_count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="шлях до Caffe 'deploy' prototxt файлу")
    ap.add_argument("-m", "--model", required=True, help="шлях до попередньо натренованої Caffe моделі")
    ap.add_argument("-c", "--confidence", type=float, default=0.6, help="мінімальний поріг достовірності")
    ap.add_argument("-v", "--video", default="Video_3.mp4",  help="шлях до відеофайлу або камера (0)")
    args = vars(ap.parse_args())

    detector = PersonDetector(args["prototxt"], args["model"], args["confidence"])

    source = 0 if args["video"] == "0" else args["video"]
    cap = cv2.VideoCapture(source)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Кінець відео або помилка читання кадру.")
            break

        frame = imutils.resize(frame, width=900)
        frame, count = detector.detect_people(frame)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"People: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Person Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
