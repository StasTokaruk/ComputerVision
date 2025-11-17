import matplotlib
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

image = cv2.imread("img1.jpg")

small = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

pixels = small.reshape((-1, 3))
pixels = np.float32(pixels)

k = 10
model = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
model.fit(pixels)
labels = model.predict(pixels)

centers = model.means_
centers = np.uint8(centers)

segmented_pixels = centers[labels]

seg_small = segmented_pixels.reshape(small.shape)
seg = cv2.resize(seg_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


plt.imshow(seg)
plt.title(f"Gaussian Mixture")
plt.axis("off")
plt.show()

