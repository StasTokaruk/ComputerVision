import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")

image = cv2.imread("img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

small = cv2.resize(image, None, fx=0.3, fy=0.3)

pixel = small.reshape((-1, 3))
pixel = np.float32(pixel)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
k = 4
_, labels, centers = cv2.kmeans(pixel, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]

seg_small = segmented.reshape(small.shape)
seg = cv2.resize(seg_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

plt.imshow(seg)
plt.title("K-Means")
plt.axis("off")
plt.show()