import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from skimage.filters import gabor, gaussian
from skimage.restoration import (denoise_tv_chambolle)
# Not: pywt, sklearn gibi kütüphaneler bu final kodda kullanılmadığı için import edilmedi.

# --- 1. ADIM: GÜRÜLTÜ TEMİZLEME (TV Denoising) ---

def apply_tv_denoising(I_input_rgb, weight=0.1):
    """
    Total Variation (TV) Denoising uygulayarak görüntüdeki gürültüyü temizler (ROF Modeli).
    Bu, Gabor'un gürültüyü keskinleştirmesini engeller.
    """
    # Görüntüyü 0-1 aralığında float'a dönüştür
    I_float = I_input_rgb.astype(np.float32) / 255.0
    
    # TV Denoising'i her kanala (R, G, B) ayrı ayrı uygula
    I_denoised_R = denoise_tv_chambolle(I_float[:, :, 0], weight=weight)
    I_denoised_G = denoise_tv_chambolle(I_float[:, :, 1], weight=weight)
    I_denoised_B = denoise_tv_chambolle(I_float[:, :, 2], weight=weight)
    
    # Kanalları birleştirme
    I_denoised_float = np.stack([I_denoised_R, I_denoised_G, I_denoised_B], axis=-1)
    
    # 0-255 aralığına geri dönüştürme ve uint8 formatına çevirme
    I_denoised_uint8 = np.clip(I_denoised_float * 255, 0, 255).astype(np.uint8)
    
    return I_denoised_uint8

# --- 2. ADIM: GABOR DOKU GÜÇLENDİRME (Yardımcı Fonksiyonlar) ---

def get_brightness(image):
    """(Gabor fonksiyonu için gereken yardımcı fonksiyon)"""
    image_rgb = image.convert('RGB')
    arr = np.asarray(image_rgb, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    bright = np.sqrt(0.299 * R ** 2 + 0.587 * G ** 2 + 0.114 * B ** 2) / 255.0
    return float(np.mean(bright))

def enhance_brightness(image):
    """(Gabor fonksiyonu için gereken yardımcı fonksiyon)"""
    mean_brightness = get_brightness(image)
    a, b = [0.3, 1]
    if mean_brightness < 0.1: a = 0.1
    min_, max_ = [0, 1]
    new_brightness = (b - a) * (mean_brightness - min_) / (max_ - min_) + a
    
    if mean_brightness <= 0:
        brightness_factor = new_brightness
    else:
        brightness_factor = new_brightness / mean_brightness

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)


# --- ANA FONKSİYON 2: GABOR DOKU GÜÇLENDİRME ---

def applygabor(image_input, orientations=8, frequencies=None, alpha=0.20, beta=0.03):
    """
    Gabor filtresi ile dokuyu güçlendirir ve Chrominance (beta) ile destekleyerek kenar parlamasını azaltır.
    
    Args:
        image_input (numpy.ndarray): Girdi görüntüsü (Denoise edilmiş olmalı).
        alpha (float): Luminans kanalına eklenen ana güç (Dokunun parlaklığı).
        beta (float): Krominans kanallarına (a, b) eklenen destek gücü (Rengin gücü).
    """
    
    # 1. Görüntüyü PIL RGB'ye çevir
    pil_initial = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
    img_rgb_initial = np.asarray(pil_initial, dtype=np.uint8)

    # 2. Opsiyonel Ön İşleme (Orijinal koddan korunan adımlar)
    pil_proc = ImageEnhance.Color(pil_initial).enhance(0.5)
    pil_proc = enhance_brightness(pil_proc)
    
    # 3. Gabor için Gri Tonlamalı Float Görüntü (0..1)
    gray = pil_proc.convert('L')
    img_arr = np.asarray(gray, dtype=np.float32) / 255.0

    # 4. Gabor Filtre Bankası Uygulaması
    if frequencies is None:
        frequencies = [0.05, 0.1, 0.2]

    mags = []
    thetas = np.linspace(0, np.pi, orientations, endpoint=False)
    for theta in thetas:
        for freq in frequencies:
            if not (0 < freq <= 0.5): continue
            real, imag = gabor(img_arr, frequency=freq, theta=theta)
            mag = np.sqrt(real ** 2 + imag ** 2)
            mags.append(mag)

    # 5. Haritayı Birleştirme (Mean) ve Yumuşatma
    agg = np.mean(np.stack(mags, axis=0), axis=0)
    agg_smoothed = gaussian(agg, sigma=0.5)

    # 6. Gabor Haritasını Normalize Etme [0, 1]
    a_min, a_max = agg_smoothed.min(), agg_smoothed.max()
    gmap = (agg_smoothed - a_min) / (a_max - a_min) if a_max - a_min > 1e-8 else np.zeros_like(agg_smoothed)

    # 7. LAB Uzayında Doku Güçlendirme (Luminans ve Krominans Desteği)
    rgb = np.asarray(pil_initial, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) 
    
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L_norm = L / 255.0 

    # Doku Güçlendirme (L_enh = L + alpha * gmap)
    L_enh = np.clip(L_norm + alpha * gmap, 0.0, 1.0)
    
    # Krominans Kanallarına Doku Desteği (beta)
    a_enh = np.clip(a + beta * gmap * 255.0, 0, 255) 
    b_enh = np.clip(b + beta * gmap * 255.0, 0, 255)

    lab_enh = lab.copy()
    lab_enh[:, :, 0] = (L_enh * 255.0)
    lab_enh[:, :, 1] = a_enh
    lab_enh[:, :, 2] = b_enh

    # 8. RGB'ye Geri Dönüştürme
    rgb_enh = cv2.cvtColor(np.clip(lab_enh, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return rgb_enh

# --- 3. ADIM: RENK CANLILIĞI (DOYGUNLUK) ---

def enhance_chrominance(image_rgb_uint8, saturation_factor=2.3):
    """
    RGB görüntünün renk canlılığını (saturation) LAB uzayında artırır.
    """
    # 1. Görüntüyü LAB uzayına çevir
    lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # 3. a ve b kanallarını ortalamadan (128) uzaklaştır
    a_enhanced = np.clip(128 + saturation_factor * (a - 128), 0, 255)
    b_enhanced = np.clip(128 + saturation_factor * (b - 128), 0, 255)
    
    # 4. Kanalları birleştir
    lab_enhanced = np.stack([L, a_enhanced, b_enhanced], axis=-1)
    
    # 5. LAB'den tekrar RGB'ye çevir
    rgb_enhanced = cv2.cvtColor(lab_enhanced.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return rgb_enhanced

# --- 4. NİHAİ PİPELINE UYGULAMASI ---

if __name__ == "__main__":
    
    # Girdi: Histogram Rescaling ve WB sonrası temizlenmiş görüntü
    sample_path = "NECI_Global_Mapped_Result.jpg" 
    
    if os.path.exists(sample_path):
        
        # 1. Görüntüyü Yükle
        img_bgr = cv2.imread(sample_path)
        img_rgb_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. GÜRÜLTÜ TEMİZLEME (TV Denoising)
        print("-> 1. Adım: Gürültü Temizleme (TV Denoising) uygulanıyor...")
        denoised_rgb = apply_tv_denoising(img_rgb_input, weight=0.1)

        # 3. DOKU GÜÇLENDİRME (Gabor)
        print("-> 2. Adım: Doku Güçlendirme (Gabor) uygulanıyor...")
        gabor_result_rgb = applygabor(denoised_rgb, 
                                      orientations=8, 
                                      frequencies=[0.05, 0.1, 0.2], 
                                      alpha=0.25, # Nazik keskinleştirme
                                      beta=0.04) 

        # 4. RENK CANLILIĞI (Vibrance)
        print("-> 3. Adım: Renk Canlılığı (Chrominance) uygulanıyor...")
        # Hata Düzeltmesi: 2.5 yerine 1.25 (Doğal artış)
        final_enhanced_rgb = enhance_chrominance(gabor_result_rgb, saturation_factor=1.8) 
        
        # 5. NİHAİ GÖRSELLEŞTİRME
        print("-> 4. Adım: Sonuçlar görselleştiriliyor...")
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img_rgb_input)
        plt.title("1. Giriş (Gürültülü)")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(denoised_rgb)
        plt.title("2. Denoising Sonrası (Yumuşak)")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(gabor_result_rgb)
        plt.title("3. Gabor Sonrası (Keskin)")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(final_enhanced_rgb)
        plt.title("4. Final Görüntü (Canlı Renkler)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        Image.fromarray(final_enhanced_rgb).save("final_colored_output.png")
        print(f"Final Görüntü kaydedildi: final_colored_output.png")
        
    else:
        print(f"Hata: Örnek resim bulunamadı: {sample_path}")