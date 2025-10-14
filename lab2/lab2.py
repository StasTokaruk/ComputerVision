import cv2
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

#Завантаження зображення
image_path = "img.png"
image = cv2.imread(image_path)
if image is None:
    print("Помилкан не вдалося завнтажити зображення")
else:
    # Зменшення розміру зображення
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    #Параметри
    alpha = 1.7
    beta = -76
    threshold_value = 193
    min_area = 230
    max_area = 14962
    aspect_ratio_min = 0
    aspect_ratio_max = 2.4
    kernel_size = 0

    #Кольрова обробка і робота з контрастом
    negative = 255 - image_rgb
    negative_gray = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.convertScaleAbs(negative_gray, alpha=alpha, beta=beta)

    #Порогова бінаризація
    _, binary = cv2.threshold(gray_eq, threshold_value, 255, cv2.THRESH_BINARY_INV)

    #Пошук контурів та фільтрація не потрібних
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings_count = 0
    final_image = image_rgb.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0: continue

            aspect_ratio = float(w) / h
            if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
                cv2.rectangle(final_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                buildings_count += 1

    #Візувлізація всіх етапів
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Рядок 1
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title("1. Оригінальне зображення")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(negative)
    axs[0, 1].set_title("2. Негатив")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(gray_eq, cmap='gray')
    axs[0, 2].set_title("3. Корекція контрасту")
    axs[0, 2].axis('off')

    # Рядок 2
    axs[1, 0].imshow(binary, cmap='gray')
    axs[1, 0].set_title("4. Порогова бінаризація")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(final_image)
    axs[1, 1].set_title(f"6. Фінальний результат ({buildings_count} будівель)")
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()