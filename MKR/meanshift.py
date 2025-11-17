import matplotlib
from sklearn.cluster import MeanShift
import cv2
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

image = cv2.imread("img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

small = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

pixels = small.reshape((-1, 3))
idx = np.random.choice(len(pixels), 10000, replace=False)
sample = pixels[idx]

ms = MeanShift(bandwidth=15, bin_seeding=True)
ms.fit(sample)

labels = ms.predict(pixels)
centers = ms.cluster_centers_.astype(np.uint8)

seg_small = centers[labels].reshape(small.shape)
seg = cv2.resize(seg_small, (image.shape[1], image.shape[0]))

plt.imshow(seg)
plt.title("MeanShift")
plt.axis("off")
plt.show()
