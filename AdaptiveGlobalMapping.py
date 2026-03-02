import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Anahtar Parametre Hesaplama Fonksiyonları (Önceki Kodunuz) ---

def calculate_image_key_value_exact(Y):
    # Makaledeki Key Value hesaplama formülü (Yalnızca Parlaklık kanalı üzerinde)
    L_normalized = (Y.astype(np.float64) / 255.0) * 100.0
    epsilon = 1e-6 
    log_L_plus_epsilon = np.log(L_normalized + epsilon)
    mean_log_L = np.sum(log_L_plus_epsilon) / Y.size
    key_value = np.exp(mean_log_L)
    return key_value

def calculate_R(key, epsilon=1e-6):
    # Yarıçap (r) hesaplama
    key = np.float64(key) 
    if key <= 50:
        r = 3 * np.log(key / 10 + epsilon)
    elif key >= 60:
        r = 3 * np.log(10 - key / 10 + epsilon)
    else:
        r = 0.0
    r_capped = np.clip(r, 0, 1.4)
    return r_capped

def calculate_x0_y0(r):
    # Çember merkezi (x0, y0) hesaplama
    if r <= np.sqrt(0.5):
        r = np.sqrt(0.5)
        
    a = 2.0
    b = -2.0
    c = 1.0 - r**2
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return (0.5, 0.5)
        
    x0_1 = (-b + np.sqrt(discriminant)) / (2 * a)
    x0_2 = (-b - np.sqrt(discriminant)) / (2 * a)

    x0 = x0_1 if x0_1 >= 0.5 else x0_2
    y0 = 1.0 - x0
    
    return (x0, y0)

# --- 2. Adaptif Global Eşleme (Denklem 3) Uygulama Fonksiyonu ---

def calculate_I_gm(key, r, y0, x0, I_orig_norm):
    """
    NECI'nin Adaptif Global Eşleme (Circular Mapping) Denklemi (3)'ü uygular.
    
    Args:
        key (float): Görüntünün Anahtar Değeri.
        r, y0, x0 (float): Çemberin sabit parametreleri.
        I_orig_norm (numpy.ndarray): Orijinal parlaklık kanalı (0-1 aralığında).
        
    Returns:
        numpy.ndarray: Global olarak eşlenmiş parlaklık kanalı (0-1 aralığında).
    """
    
    I_gm = np.zeros_like(I_orig_norm, dtype=np.float64)
    
    # NaN hatalarını önlemek için Karekök içi terimi kontrol et (pozitif olmalı)
    sqrt_term = r**2 - (I_orig_norm - x0)**2
    
    # Hata durumunda (karekök içi negatifse) oradaki tonu değiştirmemek en mantıklısıdır.
    valid_mask = sqrt_term >= 0
    I_gm[~valid_mask] = I_orig_norm[~valid_mask]
    
    sqrt_val = np.sqrt(sqrt_term[valid_mask])
    
    # -----------------------------------------------------
    # Eşleme Kurallarının Uygulanması
    # -----------------------------------------------------

    if key <= 50:
        # Aşırı Karanlık: Üst yarım daire (Aydınlatma)
        I_gm[valid_mask] = y0 + sqrt_val
        
    elif key >= 60:
        # Aşırı Aydınlık: Alt yarım daire (Kontrastı koruma/Sıkıştırma)
        I_gm[valid_mask] = y0 - sqrt_val
        
    else:
        # Normal Aralık: Değişiklik yapma
        I_gm = I_orig_norm
        
    # Sınırlandırma (0-1 aralığında tutma)
    I_gm = np.clip(I_gm, 0, 1.0)

    return I_gm

# --- 3. Ana Çalıştırma Fonksiyonu ve Görselleştirme ---

def run_nec_global_mapping(image_path):
    # 1. Görüntüyü Oku
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"HATA: '{image_path}' yolunda resim bulunamadı veya okunamadı.")
        return

    # 2. Parlaklık Kanalını Çıkar (YCrCb uzayında)
    YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y_orig = YCrCb[:,:,0]
    
    # 3. Parametreleri Hesapla
    key_value = calculate_image_key_value_exact(Y_orig)
    R_val = calculate_R(key_value)
    X0_val, Y0_val = calculate_x0_y0(R_val)
    
    print("-" * 40)
    print(f"Key Value: {key_value:.4f}")
    print(f"Çember Yarıçapı (r): {R_val:.4f}")
    print(f"Çember Merkezi (x0, y0): ({X0_val:.4f}, {Y0_val:.4f})")
    
    # 4. Global Eşlemeyi Uygula (Luminance kanalı üzerinde)
    Y_norm_orig = Y_orig.astype(np.float64) / 255.0
    Y_mapped_norm = calculate_I_gm(key_value, R_val, Y0_val, X0_val, Y_norm_orig)
    
    # 5. Eşlenmiş Parlaklığı Tekrar [0, 255] aralığına getir
    Y_mapped = np.clip(Y_mapped_norm * 255, 0, 255).astype(np.uint8)
    
    # 6. Eşlenmiş Parlaklığı Orijinal Görüntüye Yerleştir
    YCrCb_mapped = YCrCb.copy()
    YCrCb_mapped[:,:,0] = Y_mapped
    
    # 7. Sonucu BGR (Renkli) Uzaya Dönüştür
    img_mapped_bgr = cv2.cvtColor(YCrCb_mapped, cv2.COLOR_YCrCb2BGR)

    # 8. Görselleştirme
    img_orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_mapped_rgb = cv2.cvtColor(img_mapped_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig_rgb)
    plt.title(f"Orijinal Görüntü (Key={key_value:.2f})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_mapped_rgb)
    plt.title("Adaptif Global Eşleme Sonrası")
    plt.axis('off')
    cv2.imwrite("NECI_Global_Mapped_Result.jpg", img_mapped_bgr)
    
    plt.suptitle("NECI Adaptif Global Eşleme (Ön İşlem)")
    plt.show()

# --- KULLANIM ÖRNEĞİ ---
if __name__ == '__main__':
    # !!! BURAYI DÜZENLEYİN !!!
    # 'llnet_ciktim.jpg' yerine kendi LLNet/Gamma çıktınızın yolunu yazın.
    resim_yolu = "gamma_corrected_recursive_denoised.png"
    resim_yolu2="Image_Process/Image_Enhacement_Dataset/LowDataset/230.png"
    # Kodun düzgün çalışması için geçerli bir resim yolu girmeniz gerekir.
    # Örn: run_nec_global_mapping('avengers_figurlu_soluk.jpg')
    run_nec_global_mapping(resim_yolu)