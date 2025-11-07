import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import rasterio
import pandas as pd

matplotlib.use("TkAgg")

B3_PATH = 'B03.tiff'
B4_PATH = 'B04.tiff'
B8_PATH = 'B08.tiff'
B11_PATH = 'B11.tiff'

def load_band(path, name, ref_shape=None):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        meta = src.meta
        if ref_shape and (src.height != ref_shape[0] or src.width != ref_shape[1]):
            raise ValueError(f"Розмір {name} не збігається з референсом.")
    return arr, meta

def safe_index(b1, b2):
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = (b1 - b2) / (b1 + b2)
    idx[~np.isfinite(idx)] = np.nan
    return idx

try:
    B4, meta = load_band(B4_PATH, 'B4 (Red)')
    B8, _ = load_band(B8_PATH, 'B8 (NIR)', ref_shape=B4.shape)
    B11, _ = load_band(B11_PATH, 'B11 (SWIR)', ref_shape=B4.shape)
    B3, _ = load_band(B3_PATH, 'B3 (Green)', ref_shape=B4.shape)


    PIXEL_SIZE = meta['transform'][0]
    ROWS, COLS = B4.shape

    NDVI = safe_index(B8, B4)
    NDBI = safe_index(B11, B8)
    NDWI = safe_index(B3, B8)

    valid_mask = np.isfinite(NDVI) & np.isfinite(NDBI) & np.isfinite(NDWI)
    X = np.vstack([NDVI.flatten(), NDBI.flatten(), NDWI.flatten()]).T
    X_valid = X[valid_mask.flatten()]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    K = 4
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_valid = km.fit_predict(X_scaled)
    labels = np.full(B4.size, -1)
    labels[valid_mask.flatten()] = labels_valid
    classified_map = labels.reshape(B4.shape)

    centroids = pd.DataFrame(km.cluster_centers_, columns=['NDVI', 'NDBI', 'NDWI'])

    field_clusters = centroids.index[(centroids["NDBI"] > 0.2)].tolist()
    water_clusters = centroids.index[(centroids["NDWI"] > 0.1)].tolist()
    city_clusters = centroids.index[(centroids["NDVI"] > 0.3) & (centroids["NDBI"] < 0.1)].tolist()

    mask_field = np.isin(classified_map, field_clusters)
    mask_water = np.isin(classified_map, water_clusters)
    mask_city = np.isin(classified_map, city_clusters)

    def calc_area(mask):
        pix_area = PIXEL_SIZE**2
        return np.sum(mask) * pix_area / 10000

    area_field = calc_area(mask_field)
    area_water = calc_area(mask_water)
    area_city = calc_area(mask_city)

    print(f"Поля: {area_field:.2f} га")
    print(f"Вода: {area_water:.2f} га")
    print(f"Місто: {area_city:.2f} га")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(NDVI, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title('NDVI')
    plt.colorbar(label='NDVI')

    plt.subplot(1, 2, 2)
    rgb_map = np.zeros((ROWS, COLS, 3))
    rgb_map[mask_field] = [0.1, 0.7, 0.1]
    rgb_map[mask_water] = [0.1, 0.5, 1.0]
    rgb_map[mask_city] = [0.8, 0.2, 0.2]
    plt.imshow(rgb_map)
    plt.title("Карта класифікації")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"ERROR: {e}")
