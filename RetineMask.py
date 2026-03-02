import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- NECI ADIM 1: Çok Ölçekli Maske Hesaplama (L_mask) ---

def calculate_retinex_mask(L_input, min_sigma=1, divisor_k=8):
    """
    Denklem 6, 7 ve 8'e dayanarak L_mask (Çok Ölçekli Arka Plan Aydınlatması) hesaplar.
    """
    H, W = L_input.shape

    # K Değerini Hesaplama (Denklem 8)
    K = max(H, W) / divisor_k
    num_scales = int(np.floor(np.log2(K)))
    if num_scales < 1:
        num_scales = 1

    Fr_filter = np.zeros_like(L_input, dtype=np.float32)

    # Çok Ölçekli Gauss filtrelerini uygulayıp toplama
    for i in range(1, num_scales + 1):
        sigma = min_sigma * (2 ** i)

        # cv2.GaussianBlur ile konvolüsyon
        current_mask = cv2.GaussianBlur(L_input.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        Fr_filter += current_mask

    # Ortalama Alarak L_mask'ı elde etme
    L_mask = Fr_filter / num_scales

    return L_mask


# --- NECI ADIM 2: Modifiye Luminans Geliştirme (Denklem 9) ---

def enhance_luminance_retinex(L_input, L_mask, epsilon=1e-5):
    """
    Modifiye Retinex formülünü kullanarak geliştirilmiş luminansı hesaplar. (Denklem 9)

    L_retinex = L_gm / (log10(L_mask) + epsilon)

    Burada L_gm (Globally Mapped Luminance), teorik olarak bir önceki adımdır.
    Basitlik için bu örnekte L_gm olarak L_input'un kendisini kullanacağız.
    """

    # 1. Giriş Luminansını (L_gm) ayarla
    L_gm = L_input.astype(np.float32)

    # 2. Maskeye Logaritma Uygulama (L_mask'ı 10 tabanında logaritmasını alıp epsilon ekleme)
    L_mask_stabilized = L_mask.astype(np.float32) + epsilon

    # Modifiye Payda Hesaplama: log10(L_mask) + epsilon
    denominator = np.log10(L_mask_stabilized) + epsilon

    # 3. Oranlama yaparak geliştirilmiş Luminans (L_retinex) hesaplama (Denklem 9)
    # Paydanın sıfır olmamasını sağlamak için epsilon ile korunur.
    L_retinex = L_gm / denominator

    return L_retinex


def chrominance_enhancement(L_enh, L_gm, C_channel, max_chroma_gain=10.0, epsilon=1e-5):
    """
    Referans haritası (M_ref) kullanarak krominansı geliştirir (Denklemler 10, 11).

    - L_enh: Modifiye Retinex ile elde edilen luminans (2D, float).
    - L_gm: Modifiye edilmeden önceki luminans (orijinal L channel, 2D).
    - C_channel: Modifiye edilmeden önceki krominans kanalı (a veya b), 2D.
    - Dönen değer: aynı boyutta uint8 tipinde geliştirilmiş krominans kanalı.
    """
    # Hesaplamaları float yap
    L_enh_f = L_enh.astype(np.float32)
    L_gm_f = L_gm.astype(np.float32)
    C_f = C_channel.astype(np.float32)

   

    # 1. Referans Haritası Hesaplama (Denklem 10)
    # L_enh / L_gm (küçük değerler için epsilon eklenir)
    M_ref = (L_enh_f + epsilon) / (L_gm_f + epsilon)

    # 2. Renk Patlamasını Engellemek için Üst Sınır Koyma
    M_ref_clipped = np.clip(M_ref, 1.0, max_chroma_gain)

    # 3. Krominans Geliştirme (Denklem 11)
    C_enh = C_f * M_ref_clipped

    # 4. Görme aralığına geri sınırlama ve tip dönüşümü
    C_enh_clipped = np.clip(C_enh, 0, 255).astype(np.uint8)

    return C_enh_clipped


# --- UYGULAMA VE GÖRSELLEŞTİRME ---

# **Lütfen 'NECI_Global_Mapped_Result.jpg' adında düşük kontrastlı bir resim dosyasını 
# bu kodun bulunduğu klasöre yerleştirin.**
IMAGE_PATH = 'NECI_Global_Mapped_Result.jpg'

try:
    # 1. RGB Görüntüyü Yükleme
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"'{IMAGE_PATH}' dosyası bulunamadı. Lütfen bir resim yükleyin.")

    # Orijinal görüntüyü görselleştirmek için RGB'ye çevir
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. RGB'den L*a*b*'ye Dönüşüm
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # L kanalı, Luminans (Parlaklık) bilgisini içerir (0-255 aralığında)
    L_channel = img_lab[:, :, 0]

    # 3. NECI Adımlarının Uygulanması
    print("1. Adım: Çok Ölçekli Maske (L_mask) hesaplanıyor...")
    L_mask = calculate_retinex_mask(L_channel)

    print("2. Adım: Modifiye Retinex (L_retinex) uygulanıyor...")
    L_retinex_enhanced = enhance_luminance_retinex(L_channel, L_mask)

    # 3. Adım: Krominans geliştirme uygulanıyor...
    print("3. Adım: Krominans geliştirme uygulanıyor...")
    # a ve b kanallarını (modifiye edilmeden önceki) al
    a_channel_orig = img_lab[:, :, 1]
    b_channel_orig = img_lab[:, :, 2]

    enhanced_a = chrominance_enhancement(L_retinex_enhanced, L_channel, a_channel_orig)
    enhanced_b = chrominance_enhancement(L_retinex_enhanced, L_channel, b_channel_orig)

    # 4. Normalizasyon ve Yeniden Sınırlama L_retinex için
    L_retinex_normalized = cv2.normalize(L_retinex_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    L_retinex_uint8 = np.clip(L_retinex_normalized, 0, 255).astype(np.uint8)

    # 5. Geliştirilmiş L ve C kanallarını geri koyma
    img_lab_enh = img_lab.copy()
    img_lab_enh[:, :, 0] = L_retinex_uint8
    img_lab_enh[:, :, 1] = enhanced_a
    img_lab_enh[:, :, 2] = enhanced_b

    # 6. Sonucu RGB'ye Geri Çevirme
    img_enhanced_bgr = cv2.cvtColor(img_lab_enh, cv2.COLOR_LAB2BGR)
    img_enhanced_rgb = cv2.cvtColor(img_enhanced_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Luminance_Chrominance_Enhanced_Result.png", cv2.cvtColor(img_enhanced_rgb, cv2.COLOR_RGB2BGR))

    # 7. Görselleştirme: 3 Adımın Gösterimi
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Orijinal görüntü
    axes[0].imshow(img_rgb)
    axes[0].set_title('Orijinal Görüntü (RGB)')
    axes[0].axis('off')

    # L_mask görselleştirme (grayscale)
    im1 = axes[1].imshow(L_mask, cmap='gray')
    axes[1].set_title('Adım 1: Çok Ölçekli Maske (L_mask)')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Son iyileştirilmiş görüntü
    axes[2].imshow(img_enhanced_rgb)
    axes[2].set_title('Adım 3: Krominans + Luminans ile Geliştirilmiş (RGB)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    print("İşlem tamamlandı. 3 adım gösteriliyor.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")