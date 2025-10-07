import cv2
import numpy as np

img = cv2.imread('satellite_image.jpg')
if img is None:
    raise ValueError("Зображення не знайдено. Перевір шлях!")

cv2.imshow("Original Image", img)
cv2.waitKey(0)

# Перетворюємо в градації сірого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)

# Розмиття для зменшення шуму
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)

# Адаптивна бінаризація для кращого виділення світлих і темних ділянок
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 8)
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

# Морфологічні операції (закриття та відкриття)
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow("Morphology Image", morph)
cv2.waitKey(0)

# Знаходимо контури
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фільтруємо контури за площею (менше значення, щоб врахувати маленькі будинки)
min_area = 50
houses = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Малюємо контури на оригінальному зображенні
for cnt in houses:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Знайдено приблизно {len(houses)} будинків")
cv2.imshow("Detected Houses", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
