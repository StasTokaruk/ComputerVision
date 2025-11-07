import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

matplotlib.use("TkAgg")

# Сегментація зображення за допомогою алгоритму K-Means кластеризації
def kmeans_segmentation(img, k_clusters: int = 5):
    if img is None: return None
    data = img.reshape((-1, 3))
    data = np.float32(data)

    try:
        kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10, max_iter=300)
        kmeans.fit(data)
    except ValueError:
        return None

    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_img = centers[labels.flatten()].reshape(img.shape)

    masks = []
    for i in range(k_clusters):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[labels.reshape(img.shape[:2]) == i] = 255
        masks.append(mask)

    return segmented_img, labels, centers, masks


def load_and_preprocess_images(path1: str, path2: str, target_width: int = 800):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    def resize_keep_aspect(img, w):
        if img is None: return None
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if target_width:
        img1 = resize_keep_aspect(img1, target_width)

    if img2 is not None and img1 is not None:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    def apply_clahe(img):
        if img is None: return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    img1_clahe = apply_clahe(img1)
    img2_clahe = apply_clahe(img2)

    return img1_clahe, img2_clahe, img1.copy(), img2.copy()


img1, img2, img1_orig, img2_orig = load_and_preprocess_images('img_1.png', 'img_2.png', target_width=800)

segmented_img1, labels1, centers1, masks1 = kmeans_segmentation(img1, k_clusters=5)
segmented_img2, labels2, centers2, masks2 = kmeans_segmentation(img2, k_clusters=5)
print(f"K-Means сегментація: Виділено {len(centers1)} кластерів у Img 1 та {len(centers2)} у Img 2")


def detect_keypoints_on_contours(masks, approx_epsilon: float = 0.005):
    kp_list = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = approx_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                x, y = point[0]
                kp_list.append(cv2.KeyPoint(float(x), float(y), 3))

    return kp_list

kp1 = detect_keypoints_on_contours(masks1)
kp2 = detect_keypoints_on_contours(masks2)

print(f"Ключові точки: {len(kp1)} (Img 1) | {len(kp2)} (Img 2)")

sift = cv2.SIFT_create()
kp1, des1 = sift.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1)
kp2, des2 = sift.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2)

des1_shape = des1.shape
des2_shape = des2.shape
print(f"SIFT дескриптори: {des1_shape} (Img 1) | {des2_shape} (Img 2)")

def flann_match_descriptors(des1, des2, ratio: float = 0.72):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: return []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        knn_matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []
    return [m[0] for m in knn_matches if len(m) == 2 and m[0].distance < ratio * m[1].distance]


def ransac_inliers(kp1, kp2, matches, reproj_thresh: float = 4.0):
    if len(matches) < 4: return [], None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    inliers = [m for m, s in zip(matches, status.flatten()) if s == 1]
    return inliers, H


good_matches = flann_match_descriptors(des1, des2)
inliers, H = ransac_inliers(kp1, kp2, good_matches)

print(f"Good matches: {len(good_matches)}")
print(f"Інлаєри (після RANSAC): {len(inliers)}")

P_identification = len(inliers) / max(1, len(good_matches))

print(f"Ймовірність ідентифікації: {P_identification:.3f}")

def visualize_matches(img1, kp1, img2, kp2, matches, P_identification):
    vis_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"{len(matches)} інлаєрів. Ймовірність: {P_identification:.3f}")
    plt.show()


visualize_matches(img1, kp1, img2, kp2, inliers, P_identification)
