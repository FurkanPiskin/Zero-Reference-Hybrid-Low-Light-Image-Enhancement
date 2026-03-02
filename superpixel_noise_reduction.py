import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2lab
from scipy import ndimage
import os
from datetime import datetime # Sadece hata ayıklama çıktısı için

# --- ANA FONKSİYON: ADAPTIF SÜPERPİKSEL DENOISING ---

def apply_superpixel_adaptive_denoising(I_input_rgb, n_segments=1000, compactness=20, base_weight=0.04, max_factor=5.0):
    """
    Süperpiksel bazında gürültü-doku oranını (alpha) hesaplar ve bu oranın ortalamasını 
    kullanarak TV Denoising'i adapte eder. Hata giderilmiştir.
    
    Args:
        I_input_rgb (numpy.ndarray): RGB formatında girdi görüntüsü (uint8).
        n_segments (int): SLIC ile oluşturulacak yaklaşık bölge sayısı.
        compactness (float): SLIC kompaktlık derecesi.
        base_weight (float): TV Denoising için minimum ağırlık.
        max_factor (float): Ağırlığın ne kadar artırılabileceği çarpanı.
        
    Returns:
        numpy.ndarray: Adaptif olarak temizlenmiş RGB görüntü (uint8).
    """
    
    I_input_rgb = I_input_rgb.astype(np.uint8)
    I_float = img_as_float(I_input_rgb)
    I_lab = rgb2lab(I_float)
    L_channel = I_lab[:, :, 0] # Luminans kanalı (L)
    
    # 1. SLIC Segmentasyonu
    segments = slic(I_lab, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1)
    
    # 2. Gradyan (Doku) Magnitüdü Hesaplama (g_pi)
    sobel_h = ndimage.sobel(L_channel, axis=0)
    sobel_v = ndimage.sobel(L_channel, axis=1)
    gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    
    # 3. Alpha Haritası (α) Hesaplama
    alpha_map = np.zeros(L_channel.shape, dtype=np.float64)
    epsilon = 1e-6
    
    for segment_id in np.unique(segments):
        mask = (segments == segment_id)
        
        # Gürültü Seviyesi (σ_pi) ve Doku Seviyesi (g_pi)
        sigma_pi = np.std(L_channel[mask])
        g_pi = np.mean(gradient_magnitude[mask])
        
        # Alpha Oranı: α_pi = σ_pi / g_pi
        alpha_pi = sigma_pi / (g_pi + epsilon) 
        alpha_map[mask] = alpha_pi

    # 4. KRİTİK ÇÖZÜM: Adaptif Ağırlık Skalarını Hesaplama
    
    # Alpha Haritasını [0, 1] aralığına normalize et
    alpha_min, alpha_max = alpha_map.min(), alpha_map.max()
    alpha_norm = (alpha_map - alpha_min) / (alpha_max - alpha_min + epsilon)
    
    # Ortalama Alpha Değerini Bul
    mean_alpha_norm = np.mean(alpha_norm)
    
    # TV Denoising Ağırlığı (weight) = Min + (Max - Min) * Ortalama_Alpha
    max_weight = base_weight * max_factor
    adaptive_weight_scalar = base_weight + (max_weight - base_weight) * mean_alpha_norm
    
    print(f"-> Hesaplanan Adaptif Ağırlık: {adaptive_weight_scalar:.4f} (Max={max_weight:.4f})")
    
    # 5. Adaptif (Sabit Ağırlıklı) TV Denoising Uygulaması
    I_float = I_input_rgb.astype(np.float32) / 255.0
    
    # Hata giderildi: TV Denoising'e tekil (skalar) ağırlık veriliyor.
    I_denoised_R = denoise_tv_chambolle(I_float[:, :, 0], weight=adaptive_weight_scalar, channel_axis=None)
    I_denoised_G = denoise_tv_chambolle(I_float[:, :, 1], weight=adaptive_weight_scalar, channel_axis=None)
    I_denoised_B = denoise_tv_chambolle(I_float[:, :, 2], weight=adaptive_weight_scalar, channel_axis=None)

    I_denoised_float = np.stack([I_denoised_R, I_denoised_G, I_denoised_B], axis=-1)
    
    # 6. Çıktı
    I_denoised_uint8 = np.clip(I_denoised_float * 255, 0, 255).astype(np.uint8)
    
    return I_denoised_uint8

# --------------------
# KULLANIM VE GÖRSELLEŞTİRME
# --------------------

IMAGE_PATH = 'Robust_WB_output.png' # Lütfen doğru resim yolunu girin

try:
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Hata: Resim dosyası bulunamadı. Lütfen '{IMAGE_PATH}' yolunu kontrol edin.")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Adaptif Denoising Uygulaması
    start_time = datetime.now()
    img_denoised_adaptive = apply_superpixel_adaptive_denoising(img_rgb, 
                                                                n_segments=1000, 
                                                                compactness=20,
                                                                base_weight=0.03 , 
                                                                max_factor=2.0) 
    end_time = datetime.now()
    
    # Görselleştirme
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("1. Orijinal Giriş (Gürültülü)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_denoised_adaptive)
    plt.title("2. Adaptif Süperpiksel Denoising Sonrası")
    plt.axis("off")

    cv2.imwrite("Superpixel_Adaptive_Denoised_Result.png", cv2.cvtColor(img_denoised_adaptive, cv2.COLOR_RGB2BGR))
    
    plt.tight_layout()
    plt.show()

    print(f"\nİşlem Süresi: {(end_time - start_time).total_seconds():.2f} saniye.")
    print("Hata giderildi. Adaptif TV Denoising artık skalar ağırlıkla çalışmaktadır.")

except FileNotFoundError as e:
    print(f"\nHata: {e}")
except Exception as e:
    print(f"\nBeklenmedik bir hata oluştu: {e}")