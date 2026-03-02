import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.special import gamma as sp_gamma # Gamma fonksiyonu için

# --- 1. TONALITE DÜZELTME (RCDF & SMMS) FONKSİYONU ---

def apply_rcdf_smms_to_value(img_hsv_input, delta_value=5.0):
    """
    RCDF ve SMMS işlemlerini sadece V (Value/Parlaklık) kanalına uygular.
    Bu, görüntünün tonalite (ton dağılımı) sorununu çözer.
    """
    # Girişin HSV olduğundan emin ol
    H, S, V_orig = cv2.split(img_hsv_input)

    # 1. RCDF Uygulaması (V = (1 - exp(-V))^delta)
    V_float = V_orig.astype(np.float32) / 255.0
    
    # RCDF Formülü
    v_toned = np.power((1.0 - np.exp(-V_float)), delta_value)
    
    # 2. İstatistiksel Min-Max Ölçekleme (SMMS) - Kontrast Germe
    Vn = v_toned.min()
    Vx = v_toned.max()
    
    # SMMS Formülü: V_final = (v_toned - Vn) / (Vx - Vn)
    if (Vx - Vn) == 0:
        V_final_norm = v_toned
    else:
        V_final_norm = (v_toned - Vn) / (Vx - Vn)

    # 3. Final Ölçekleme (0.0-1.0 -> 0-255 uint8)
    V_final_uint8 = np.clip(V_final_norm * 255.0, 0, 255).astype(np.uint8)

    # H, S, ve yeni V kanalını birleştir
    # KRİTİK DÜZELTME: Tüm kanalların uint8 olduğundan emin olmak için zorunlu dönüşüm
    enhanced_hsv_final = cv2.merge([
        H.astype(np.uint8), 
        S.astype(np.uint8), 
        V_final_uint8.astype(np.uint8)
    ])
    return enhanced_hsv_final

# --- 2. TINT INTENSIFICATION (TI) ANA FONKSİYONU ---

def apply_tint_intensification(image_rgb_input, gamma_galt=0.7, delta_rcdf=5.0):
    """
    TI algoritmasının S ve V kanallarını işler ve RCDF/SMMS ile tonaliteyi düzeltir.
    """
    
    # Veri Hazırlığı
    # Girişin UINT8 olduğundan emin ol (cv2.cvtColor uint8 giriş ister)
    image_rgb_uint8 = np.clip(image_rgb_input, 0, 255).astype(np.uint8) 
    
    image_hsv = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2HSV)
    h, s_orig, v_orig = cv2.split(image_hsv)
    
    # Kanalların float (0-1) versiyonları
    s_float = s_orig.astype(np.float32) / 255.0
    v_float = v_orig.astype(np.float32) / 255.0

    # --- A. S KANALININ İŞLENMESİ (GALT) ---
    s_log = np.log(1 + s_float)
    s_galt = np.power(s_log, gamma_galt) 

    s_min, s_max = s_galt.min(), s_galt.max()
    s_galt_normalized = (s_galt - s_min) / (s_max - s_min) 
    new_s_uint8 = np.clip(s_galt_normalized * 255.0, 0, 255).astype(np.uint8)
    
    # -----------------------------------------------------
    
    # --- B. V KANALININ İŞLENMESİ (Taylor + S-Eğrisi) ---
    v1_taylor = v_float - (np.power(v_float, 3) / 6.0)
    v2_s_curve = v1_taylor / np.sqrt(1.0 + np.power(v1_taylor, 2))
    
    v_sonuc_min, v_sonuc_max = v2_s_curve.min(), v2_s_curve.max()
    v_sonuc_normalized = (v2_s_curve - v_sonuc_min) / (v_sonuc_max - v_sonuc_min)
    new_v_uint8 = np.clip(v_sonuc_normalized * 255.0, 0, 255).astype(np.uint8)
    
    # --- ARA ÇIKTI 1: Sadece S Düzeltilmiş ---
    enhanced_hsv_s_only = cv2.merge([h.astype(np.uint8), new_s_uint8, v_orig.astype(np.uint8)])
    enhanced_rgb_s_only = cv2.cvtColor(enhanced_hsv_s_only, cv2.COLOR_HSV2RGB)

    # --- ARA ÇIKTI 2: S + V Düzeltilmiş (RCDF/SMMS Girdisi) ---
    # enhanced_rgb2: S ve V işlemleri bitmiş görüntünün RGB hali.
    enhanced_hsv2 = cv2.merge([h.astype(np.uint8), new_s_uint8, new_v_uint8])
    enhanced_rgb2 = cv2.cvtColor(enhanced_hsv2, cv2.COLOR_HSV2RGB)

    # --- C. FİNAL Tİ ADIMI: TONALİTE DÜZELTMESİ (RCDF & SMMS) ---
    # apply_rcdf_smms_to_value RGB istiyor, bu yüzden enhanced_rgb2'yi HSV'ye çevirip veriyoruz.
    final_hsv_input_for_rcdf = cv2.cvtColor(enhanced_rgb2, cv2.COLOR_RGB2HSV)
    final_hsv_output = apply_rcdf_smms_to_value(final_hsv_input_for_rcdf, delta_value=delta_rcdf)
    
    final_toned_rgb = cv2.cvtColor(final_hsv_output, cv2.COLOR_HSV2RGB)
    
    return image_rgb_input, enhanced_rgb_s_only, enhanced_rgb2, final_toned_rgb

# --- TEST VE GÖRSELLEŞTİRME ---

if __name__ == "__main__":
    IMAGE_PATH = "Robust_WB_output.png" 
    
    if os.path.exists(IMAGE_PATH):
        img_bgr = cv2.imread(IMAGE_PATH)
        img_rgb_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # TI Algoritması Uygulaması
        # Delta=5.0 (Tonalite Gücü) ve gamma=0.7 (Doygunluk Gücü)
        orig_rgb, s_only_rgb, sv_processed_rgb, final_toned_rgb = \
            apply_tint_intensification(img_rgb_input, gamma_galt=0.9, delta_rcdf=6.0)

        # Görselleştirme
        plt.figure(figsize=(24, 6))

        plt.subplot(1, 4, 1)
        plt.imshow(orig_rgb)
        plt.title('1. Orijinal Giriş')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(s_only_rgb)
        plt.title('2. Sadece S Düzeltmesi (GALT)')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(sv_processed_rgb)
        plt.title('3. S + V İşlemesi (RCDF/SMMS Öncesi)')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(final_toned_rgb)
        plt.title('4. Nihai Tonalite Düzeltmesi (RCDF/SMMS)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Hata: Örnek resim bulunamadı: {IMAGE_PATH}")