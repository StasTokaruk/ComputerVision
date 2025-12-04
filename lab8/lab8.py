import numpy as np
import cv2
import open3d as o3d

IMG_PATHS = ['im2.ppm', 'im4.ppm', 'im6.ppm']
SHIFT_X = 0  # Зміщення для накладання

# Налаштування стерео (SGBM)
MIN_DISP = 0
NUM_DISP = 160  # Діапазон пошуку глибини
BLOCK_SIZE = 5

#Обчислює карту глибини між двома зображеннями.
def compute_disparity(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE ** 2,
        P2=32 * 3 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    return stereo.compute(grayL, grayR).astype(np.float32) / 16.0

#Перетворює карту глибини у хмару точок
def generate_points(img, disp):
    h, w = img.shape[:2]
    f = 0.8 * w

    # Матриця Q визначає, як 2D пікселі проектуються у 3D простір
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # пертворення масивів у список точок (N, 3)
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    disp = disp.reshape(-1)

    # прибираємо точки з помилковою глибиною
    mask = (disp > disp.min()) & (points[:, 2] < 8000) & (points[:, 2] > -5000)

    return points[mask], colors[mask]


if __name__ == '__main__':
    print("Завантаження зображень")
    # cv2.IMREAD_COLOR гарантує, що ppm прочитається коректно
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in IMG_PATHS]
    img1, img2, img3 = images

    print("Обробка стереопар")
    # Пара 1: Ліва + Центр
    disp1 = compute_disparity(img1, img2)
    p1, c1 = generate_points(img1, disp1)

    # Пара 2: Центр + Права
    disp2 = compute_disparity(img2, img3)
    p2, c2 = generate_points(img2, disp2)

    # Застосування ручного зміщення до другої пари (при тестах була потрібна)
    p2[:, 0] += SHIFT_X

    # Об'єднання двох хмар в одну
    all_p = np.vstack((p1, p2))
    all_c = np.vstack((c1, c2))

    # Створення об'єкта Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_p)
    pcd.colors = o3d.utility.Vector3dVector(all_c / 255.0)  # Нормалізація кольору

    print("Очищення від шумів")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)

    print("Центрування моделі")
    center = pcd.get_center()
    pcd.translate(-center)

    print("Візуалізація")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Лаб. робота 8 Токарюк Станіслав", width=1000, height=800)

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()