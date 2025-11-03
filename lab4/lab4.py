import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
def load_and_preprocess_images(path1: str, path2: str, target_width: int = 800):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Функція масштабування
    def resize_keep_aspect(img, w):
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if target_width:
        img1 = resize_keep_aspect(img1, target_width)

    # Приведення img2 до розміру img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    def apply_clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    img1_clahe = apply_clahe(img1)
    img2_clahe = apply_clahe(img2)

    return img1_clahe, img2_clahe, img1.copy(), img2.copy()  # копії повертаю для відображення

img1, img2, img1_orig, img2_orig = load_and_preprocess_images('img_1.png', 'img_2.png', target_width=800)

def detect_harris_corners(img, threshold: float = 0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    keypoints = np.argwhere(dst > threshold * dst.max())

    kp_list = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in keypoints]
    return kp_list

# Виявляю кути харіса та поверстаю списком об'єктів cv2.KeyPoint
kp1 = detect_harris_corners(img1)
kp2 = detect_harris_corners(img2)
print(f"Кути Харріса: {len(kp1)} (Img 1) | {len(kp2)} (Img 2)")

sift = cv2.SIFT_create()
# Обчислюємо SIFT дескриптори лише для точок, знайдених харрісом
kp1, des1 = sift.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1)
kp2, des2 = sift.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2)

des1_shape = des1.shape if des1 is not None else (0, 0)
des2_shape = des2.shape if des2 is not None else (0, 0)
print(f"SIFT дескриптори: {des1_shape} (Img 1) | {des2_shape} (Img 2)")


def flann_match_descriptors(des1, des2, ratio: float = 0.75):
    if des1 is None or des2 is None:
        return []

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Пошук найближчих сусідів
    knn_matches = flann.knnMatch(des1, des2, k=2)

    # Критерій Ратіо
    good_matches = [m[0] for m in knn_matches if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    return good_matches

#Порівнюю дескриптори за допомогою FlANN та фільтрую критерієм Ратіо
good_matches = flann_match_descriptors(des1, des2)
print(f"Good matches: {len(good_matches)}")


def ransac_inliers(kp1, kp2, matches, reproj_thresh: float = 4.0):

    if len(matches) < 4:
        return [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)

    inliers = [m for m, s in zip(matches, status.flatten()) if s == 1]
    return inliers, H

#Фільтрую інлаєри за RANSAC
inliers, H = ransac_inliers(kp1, kp2, good_matches)
print(f"Інлаєри (після RANSAC): {len(inliers)}")


P_identification = len(inliers) / max(1, len(good_matches))

print(f"Ймовірність ідентифікації: {P_identification:.3f}")

def visualize_matches(img1, kp1, img2, kp2, matches):
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Надійні Інлаєри ({len(matches)} збігів). Ймовірність: {P_identification:.3f}")
    plt.axis('off')
    plt.show()


visualize_matches(img1, kp1, img2, kp2, inliers)