import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_smart_unsharp_masking(image_rgb_uint8, radius=3, amount=0.5, threshold=3):
    """
    LAB renk uzayını kullanarak sadece Aydınlık (L) kanalını keskinleştirir.
    Renk gürültüsünü (Kırmızı noktaları) artırmadan netlik sağlar.
    """
    
    # 1. Görüntüyü RGB'den LAB formatına çevir
    # L: Lightness (Aydınlık/Detay) -> Keskinleştireceğimiz yer
    # A: Green-Red (Renk) -> Gürültünün olduğu yer (Dokunmayacağız)
    # B: Blue-Yellow (Renk) -> Gürültünün olduğu yer (Dokunmayacağız)
    image_lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB)
    
    # Kanalları ayır
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    
    # --- SADECE L KANALINI KESKİNLEŞTİR ---
    
    # İşlemler için float32'ye çevir
    l_float = l_channel.astype(np.float32) / 255.0
    
    # Bulanıklaştır
    blurred_l = cv2.GaussianBlur(l_float, (radius*2 + 1, radius*2 + 1), radius)
    
    # Maskeyi oluştur
    mask = l_float - blurred_l
    
    # Threshold uygula (Gürültüyü keskinleştirmemek için)
    if threshold > 0:
        # Threshold 0-255 aralığına göre verilir, biz float 0-1 çalışıyoruz
        thresh_float = threshold / 255.0
        mask[np.abs(mask) < thresh_float] = 0
        
    # Keskinleştirme formülü: Original + Amount * Mask
    sharpened_l = l_float + amount * mask
    
    # Kırp ve uint8'e geri çevir
    sharpened_l = np.clip(sharpened_l, 0, 1)
    sharpened_l_uint8 = (sharpened_l * 255).astype(np.uint8)
    
    # --- OPSİYONEL: RENK GÜRÜLTÜSÜNÜ TEMİZLE ---
    # A ve B kanallarındaki o kırmızı noktacıkları silmek için çok hafif blur atıyoruz
    # Bu detayları bozmaz, çünkü detay L kanalında kaldı.
    a_channel = cv2.GaussianBlur(a_channel, (3, 3), 1)
    b_channel = cv2.GaussianBlur(b_channel, (3, 3), 1)
    
    # --- BİRLEŞTİRME ---
    # İşlenmiş L kanalı ile (hafif temizlenmiş) A ve B kanallarını birleştir
    merged_lab = cv2.merge((sharpened_l_uint8, a_channel, b_channel))
    
    # Tekrar RGB'ye çevir
    result_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return result_rgb

# --- KULLANIM ---

if __name__ == "__main__":
    image_path = "Luminance_Chrominance_Enhanced_Result.png" 

    if os.path.exists(image_path):
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Eski RGB methodu (Karşılaştırma için)
        # sharpened_rgb = apply_unsharp_masking(img_rgb, radius=3, amount=1.0, threshold=3)
        
        # YENİ LAB methodu
        sharpened_smart = apply_smart_unsharp_masking(img_rgb, radius=3, amount=0.25, threshold=2)

        # Görselleştirme
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Orijinal")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(sharpened_smart)
        plt.title("Smart Sharpening (Sadece L Kanalı)")
        plt.axis("off")
        
        plt.show()
        
        cv2.imwrite("smart_sharpened_output.png", cv2.cvtColor(sharpened_smart, cv2.COLOR_RGB2BGR))
        print("Kaydedildi.")