import matplotlib
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

image = cv2.imread("img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

small = cv2.resize(image, None, fx=0.3, fy=0.3)
pixels = small.reshape((-1, 3))

k = 4
model = MiniBatchKMeans(n_clusters=k, batch_size=512)
labels = model.fit_predict(pixels)

colors = model.cluster_centers_.astype(np.uint8)
seg_small = colors[labels].reshape(small.shape)

seg = cv2.resize(seg_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

plt.imshow(seg)
plt.title("MiniBatch K-Means")
plt.axis("off")
plt.show()
