import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

#Параметри
h_min, h_max = 0, 58
s_min, s_max = 22, 168
v_min, v_max = 40, 255

block_size = 201
C = 0
min_area = 399
max_area = 50000
aspect_ratio_min = 0.24
aspect_ratio_max = 5.0

if block_size % 2 == 0:
    block_size += 1

image_path = 'img.png'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Зображення не знайдено: {image_path}")

# Конвертація у формат RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image_rgb, dim, interpolation=cv2.INTER_AREA)

#Корекція кольору (HSV)
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
inverted_mask = cv2.bitwise_not(color_mask)
masked_image = cv2.bitwise_and(resized_image, resized_image, mask=inverted_mask)

#Перетворення у відтінки сірого і розмиття
gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

#Адаптивна бінаризація
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
)

#Пошук і фільтрація контурів
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_image = resized_image.copy()
buildings_count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        continue
    aspect_ratio = float(w) / h

    if min_area < area < max_area and aspect_ratio_min < aspect_ratio < aspect_ratio_max:
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        buildings_count += 1

print(f"Знайдено будинків: {buildings_count}")

#Візуалізація результатів
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(resized_image)
axes[0, 0].set_title('1. Оригінальне зображення', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(inverted_mask, cmap='gray')
axes[0, 1].set_title('2. Інвертована маска кольорів', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(masked_image)
axes[0, 2].set_title('3. Зображення з маскою', fontsize=12)
axes[0, 2].axis('off')

axes[1, 0].imshow(blurred, cmap='gray')
axes[1, 0].set_title('4. Відтінки сірого та розмиття', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(binary, cmap='gray')
axes[1, 1].set_title('5. Адаптивна бінаризація', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(final_image)
axes[1, 2].set_title(f'6. Остаточний результат: знайдено {buildings_count} будівель',fontsize=12, fontweight='bold')
axes[1, 2].axis('off')


plt.tight_layout()
plt.show()
