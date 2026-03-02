import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

# --------------------
# Domain Transform Recursive Filter
# --------------------
def domain_transform_recursive_filter(I, sigma_s=30, sigma_r=0.1, iterations=2):
    a = np.exp(-np.sqrt(2) / sigma_s)
    G = I.copy()
    for _ in range(iterations):
        # X yönü
        for y in range(G.shape[0]):
            for x in range(1, G.shape[1]):
                w = np.exp(-abs(G[y, x] - G[y, x-1]) / sigma_r)
                G[y, x] = (1 - a*w) * G[y, x] + a*w * G[y, x-1]
        # Y yönü
        for x in range(G.shape[1]):
            for y in range(1, G.shape[0]):
                w = np.exp(-abs(G[y, x] - G[y-1, x]) / sigma_r)
                G[y, x] = (1 - a*w) * G[y, x] + a*w * G[y-1, x]
    return G

# --------------------
# Gamma fonksiyonları
# --------------------
def compute_gamma_map(I, E, epsilon=1e-6):
    gamma_map = np.abs(np.log(E + epsilon)) / (np.abs(np.log(I + epsilon)) + epsilon)
    return gamma_map

def apply_gamma_correction(I, gamma_map):
    return np.power(I, gamma_map)

# --------------------
# Görüntüleri yükle
# --------------------
I_orig = cv2.imread("Image_Process/Image_Enhacement_Dataset/LowDataset/42.png")
I_orig = cv2.cvtColor(I_orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

E_img = cv2.imread("enhanced_image2.png")
#E_img = cv2.imread("Luminance_Chrominance_Enhanced_Result.png")
E_img = cv2.cvtColor(E_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# --------------------
# 1️⃣ Gamma düzeltmesi
# --------------------
gamma_map = compute_gamma_map(I_orig, E_img)
I_gamma = apply_gamma_correction(I_orig, gamma_map)

# --------------------
# 2️⃣ Recursive filter (Gamma map optimizasyonu)
# --------------------
print("Domain Transform Recursive Filter uygulanıyor...")
gamma_map_opt = domain_transform_recursive_filter(gamma_map)
I_gamma_opt = apply_gamma_correction(I_orig, gamma_map_opt)

# --------------------
# 3️⃣ Gürültü giderme (BTV benzeri TV filtresi)
# --------------------
print("Total Variation (BTV benzeri) gürültü giderme uygulanıyor...")
lambda_val = 0.02  # filtreleme şiddeti
I_denoised = denoise_tv_chambolle(I_gamma_opt, weight=lambda_val, channel_axis=-1)
# Eğer skimage eski sürümse 'multichannel=True' kullan.

# --------------------
# Sonuçları kaydet
# --------------------
cv2.imwrite("gamma_corrected_recursive_denoised.png", cv2.cvtColor((I_denoised * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# --------------------
# Karşılaştırmalı görselleştirme
# --------------------
titles = ["Orijinal", "Gamma + Recursive Filter", "Gamma + Recursive + Gürültü Giderme"]
images = [I_orig, I_gamma_opt, I_denoised]

plt.figure(figsize=(18, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(np.clip(images[i], 0, 1))
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
