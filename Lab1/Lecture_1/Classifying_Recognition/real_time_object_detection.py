# -------------- 2 ПРИКЛАД розпізнавання обєктів в реальному масштабі часу в opencv ------------------

'''

Реалізація процесу ідентифікації об'єктів в потоковому відео
Ідентифікація здійснюється з використанням штуйної нейронної мережі
opencv-python      3.4.18.65

'''


from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# конструктор аргументів для результатів навчання нейронної мережі в протоколі prototxt
# -prototxt : шлях до prototxt Caffe файлу.
# --model : шлях до заздалегіть підготовленої моделі.
# --confidence : мінімальний поріг валідності для розпізнавання обєктів (за замовченням - 20%).
# налаштування компілятора:
# -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# ініціалізація класу списку розпізнавання обєктів та атрибутивів кольору сегменту
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# завантаження вихідних даних моделі нейронної мережі з протоколу prototxt
print("[INFO] loading model...")
'''
cv2.dnn.readNetFromCaffe - Deep Neural Network module - багатошарова ("глибока") нейромережа
https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
 '''
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# ініціалізація відеопотоку для розпізнавання
# джерело відеопотоку - файл
cap = cv2.VideoCapture('Video_0.mp4')

# джерело відеопотоку - камера
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# часова синхранізація частоти кадре відеопотоку та ініціалізації роботи алгоритму розпізнавання

fps = FPS().start()

# цикл за кадрами відеопотоку
while True:

	time.sleep(0.01)
	# зміна розміру кадра потокового відео
	_, frame = cap.read()
	frame = imutils.resize(frame, width=900)

	# параметри рамки для виділення обєкту, узгоджені з розміром кадра
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)),
		0.007843, (400, 400), 127.5)

	# аналіз обєкта в нейромережі
	net.setInput(blob)
	detections = net.forward()

	# цикл виявлення групи обєктів в кадрі
	for i in np.arange(0, detections.shape[2]):
		# отримання ймовірності розпізнавання
		confidence = detections[0, 0, i, 2]

		# селекція за ймовірністю виявлення
		if confidence > args["confidence"]:
			# обчислення координат виявленого обєкту
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# контур виявленого обєкту
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# відображення вихідного кадру з розпізнаним обєктом
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# вихід з програми за клавішою q
	if key == ord("q"):
		break
	# оновлення лічильника виявлених обєктів
	fps.update()

# зупинка таймеру кадрів та відображення поточних часових параметрів розпізнавання
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# закриття графічного вікна
cv2.destroyAllWindows()
cap.release()