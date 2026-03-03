import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
# Diğer importların altına bunu ekle:

# --- PATH INJECTION (SİSTEM YOLU ENTEGRASYONU) ---
# Şu anki dosyanın bulunduğu klasörü bul (Cubic-LLNet)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Bir üst klasöre çık (AnaProje)
parent_dir = os.path.dirname(current_dir)
# Bu üst klasörü Python'un arama yoluna (sys.path) ekle
sys.path.append(parent_dir)

# Kendi model dosyanı import ediyorsun 
# (İleride projenin adına uygun olarak LLE_Net ismini LCE_Net olarak değiştirebilirsin)
from LLE_NET import LLE_Net
from skimage.restoration import denoise_tv_chambolle
# Gamma dosyanın adının adaptive_gamma_module.py olduğunu varsayıyorum:
from GammaMap import compute_gamma_map, apply_gamma_correction, domain_transform_recursive_filter
from RetineMask import calculate_retinex_mask, enhance_luminance_retinex, chrominance_enhancement
from gabor_texture_enhancement_pipeline import applygabor, enhance_chrominance
from hsv_tonal_enhancement import apply_tint_intensification
# =====================================================================
# MODÜL 1: DERİN ÖĞRENME (LCE-Net)
# =====================================================================

def load_ai_model(weights_path, device):
    """
    Yapay zeka modelini belleğe (GPU/CPU) BİR KEZ yükler.
    Bu, C# WPF arayüzüne bağlandığında 'Cold Start' gecikmesini önler.
    """
    print("-> [1/6] Yapay Zeka Modeli Yükleniyor (LCE-Net)...")
    model = LLE_Net().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def apply_lce_net(model, image_path, device):
    """
    Orijinal görüntüyü modele sokar ve baz aydınlatmayı (Curve Estimation) yapar.
    Çıktı olarak 0.0 - 1.0 aralığında RGB Numpy Array döner.
    """
    print("-> [2/6] LCE-Net ile temel aydınlatma uygulanıyor...")
    
    # Görüntüyü yükle (Strictly RGB)
    orig_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_pil).astype(np.float32) / 255.0
    
    # Tensöre çevir ve modele gönder
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(orig_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Senin modelinin 4. çıktısı enh_final'i alıyoruz
        _, _, _, enh_final, _ = model(input_tensor)
        
    # Çıktıyı Numpy RGB formatına (0-1 aralığı) çevir
    enh_final_np = enh_final.squeeze(0).cpu().permute(1, 2, 0).numpy()
    enh_final_np = np.clip(enh_final_np, 0.0, 1.0)
    
    return orig_np, enh_final_np


def apply_gamma_and_denoise(orig_img, lce_output):
    """
    LCE-Net'ten gelen baz aydınlatmayı, Gamma haritası ve TV Denoising ile optimize eder.
    Her adımın çıktısını döndürür ki görselleştirebilelim.
    """
    print("-> [3/6] Spatial Gamma Optimizasyonu uygulanıyor...")
    
    # 1. Gamma Haritasını Çıkar
    gamma_map = compute_gamma_map(orig_img, lce_output)
    
    # 2. Haritayı Optimize Et (Recursive Filter)
    gamma_map_opt = domain_transform_recursive_filter(gamma_map, sigma_s=30, sigma_r=0.1, iterations=2)
    
    # 3. Optimize Haritayı Orijinal Görüntüye Uygula
    gamma_opt_output = apply_gamma_correction(orig_img, gamma_map_opt)
    
    print("-> [4/6] TV Denoising (Gürültü Giderme) uygulanıyor...")
    # 4. Gürültü Giderme (Şimdilik sabit lambda_val = 0.02. İleride SLIC entegre edilebilir)
    denoised_output = denoise_tv_chambolle(gamma_opt_output, weight=0.02, channel_axis=-1)
    
    return gamma_opt_output, denoised_output

# =====================================================================
def apply_neci_color_correction(denoised_img):
    """
    Gürültüden arındırılmış görüntüyü alır, LAB uzayına geçirir ve Retinex uygular.
    Pipeline standardını korumak için float-uint8 dönüşümlerini otonom yönetir.
    """
    print("-> [5/6] NECI Retinex Renk ve Kontrast Kurtarma uygulanıyor...")
    
    # 1. Pipeline (float 0-1) formatını NECI (uint8 0-255) formatına dönüştür
    img_uint8 = (np.clip(denoised_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # 2. RGB'den LAB uzayına geç (Dikkat: Görüntümüz BGR değil RGB)
    img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    L_channel = img_lab[:, :, 0]
    a_channel = img_lab[:, :, 1]
    b_channel = img_lab[:, :, 2]
    
    # 3. NECI Matematiksel İşlemleri
    L_mask = calculate_retinex_mask(L_channel)
    L_retinex = enhance_luminance_retinex(L_channel, L_mask)
    
    enh_a = chrominance_enhancement(L_retinex, L_channel, a_channel)
    enh_b = chrominance_enhancement(L_retinex, L_channel, b_channel)
    
    # 4. L kanalını 0-255 aralığına güvenli şekilde geri oturt
    L_retinex_norm = cv2.normalize(L_retinex, None, 0, 255, cv2.NORM_MINMAX)
    L_retinex_uint8 = np.clip(L_retinex_norm, 0, 255).astype(np.uint8)
    
    # 5. Kanalları birleştir ve LAB'dan RGB'ye geri dön
    img_lab_enh = cv2.merge([L_retinex_uint8, enh_a, enh_b])
    img_rgb_enh = cv2.cvtColor(img_lab_enh, cv2.COLOR_LAB2RGB)
    
    # 6. Pipeline standardı olan 0.0 - 1.0 float formatına geri çevir ve fırlat
    return img_rgb_enh.astype(np.float32) / 255.0

# =====================================================================
# MODÜL 4: GABOR DOKU GÜÇLENDİRME VE RENK CANLILIĞI
# =====================================================================
def apply_texture_and_vibrance(denoised_img):
    """
    TV Denoising'den çıkan 'yumuşatılmış' görüntünün dokusunu (texture) Gabor ile
    geri getirir ve opsiyonel olarak renk doygunluğunu artırır.
    """
    print("-> [6/6] Gabor Doku Güçlendirme (Texture) uygulanıyor...")
    
    # 1. Pipeline (float 0-1) formatını Gabor modülüne uygun (uint8 0-255) formata çevir
    img_uint8 = (np.clip(denoised_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # 2. Gabor filtresini uygula (Otonom parametrelerle)
    gabor_output = applygabor(
        img_uint8, 
        orientations=8, 
        frequencies=[0.05, 0.1, 0.2], 
        alpha=0.10,  # Doku (Luminance) şiddeti
        beta=0.04    # Renk (Chrominance) destek şiddeti
    )
    
    # 3. İsteğe bağlı olarak renk canlılığını artır
    print("-> [7/7] Renk Canlılığı (Vibrance) uygulanıyor...")
    vibrant_output = enhance_chrominance(gabor_output, saturation_factor=1.5)
    
    # 4. Pipeline standardı olan 0.0 - 1.0 float formatına geri çevir ve fırlat
    return gabor_output.astype(np.float32) / 255.0

def apply_final_polish(neci_img, sat_factor=2.35):
    """NECI'den çıkan soluk renkleri, doğal sınırları aşmadan canlandırır."""
    print("-> [Final] Renk Canlılığı (Vibrance) Cilası Atılıyor...")
    # Pipeline formatından (float) OpenCV formatına (uint8) geçiş
    img_uint8 = (np.clip(neci_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Senin yazdığın Chrominance fonksiyonu ile LAB uzayında renkleri canlandır
    vibrant_img = enhance_chrominance(img_uint8, saturation_factor=sat_factor)
    
    # Tekrar float formatına çevir ve fırlat
    return vibrant_img.astype(np.float32) / 255.0


# =====================================================================
# MODÜL 7 (TEST): AĞIR TİNT INTENSIFICATION (TI)
# =====================================================================
def apply_heavy_ti_polish(neci_img):
    """
    NECI'den çıkan resme ağır bir logaritmik renk ve ton eşlemesi uygular.
    Bu adım sadece test amaçlıdır, sonucun aşırı patlaması beklenmektedir.
    """
    print("-> [Test] Ağır TI (Tint Intensification) Uygulanıyor...")
    
    # 1. Float'tan Uint8'e geçiş
    img_uint8 = (np.clip(neci_img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # 2. Ağır TI kodunu çağır (Gamma=0.7, Delta=5.0 gibi masum görünen ama güçlü değerlerle)
    _, _, _, final_ti_rgb = apply_tint_intensification(
        img_uint8, 
        gamma_galt=0.7, 
        delta_rcdf=5.0
    )
    
    # 3. Pipeline standardına geri dön
    return final_ti_rgb.astype(np.float32) / 255.0

# =====================================================================
# ANA ORKESTRATÖR (PIPELINE ENGINE)
# =====================================================================

def run_hybrid_pipeline(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SİSTEM BAŞLATILDI (Cihaz: {device}) ---")
    
    pipeline_results = {}
    
    # ---------------------------------------------------------
    # ADIM 1: AI Tabanlı Temel Aydınlatma (LCE-Net)
    # ---------------------------------------------------------
    lce_model = load_ai_model(model_weights_path, device)
    orig_img, lce_output = apply_lce_net(lce_model, image_path, device)
    
    pipeline_results["1. Orijinal"] = orig_img
    pipeline_results["2. LCE-Net (Aydınlatma)"] = lce_output
    
    # ---------------------------------------------------------
    # ADIM 2 & 3: GAMMA + RECURSIVE FILTER & DENOISING
    # ---------------------------------------------------------
    gamma_out, denoised_out = apply_gamma_and_denoise(orig_img, lce_output)
    
    pipeline_results["3. Gamma + Recursive"] = gamma_out
    pipeline_results["4. TV Denoising"] = denoised_out
    
    # ---------------------------------------------------------
    # ADIM 4: GABOR TEXTURE (Buraya eklenecek)
    # ---------------------------------------------------------
    texture_out = apply_texture_and_vibrance(denoised_out)
    pipeline_results["4. Gabor Texture (Keskin)"] = texture_out
    # ---------------------------------------------------------
    # ADIM 5: NECI COLOR ENHANCEMENT (Buraya eklenecek)
    # ---------------------------------------------------------
    neci_out = apply_neci_color_correction(texture_out)
    pipeline_results["5. NECI "] = neci_out

    # ADIM 6: FİNAL CİLASI (Renk Canlılığı)
    # ---------------------------------------------------------
    final_out = apply_final_polish(neci_out, sat_factor=1.35)
    pipeline_results["6. Final Renk Cilası (Canlı)"] = final_out

    # ADIM 7: AĞIR TI (Senin Test Kodun)
    # ---------------------------------------------------------
    #heavy_ti_out = apply_heavy_ti_polish(neci_out)
    #pipeline_results["7. Ağır Renk Cilası (Test)"] = heavy_ti_out
    
    # =====================================================================
    # GÖRSELLEŞTİRME (VİTRİN)
    # =====================================================================
    print("-> İşlem tamamlandı. Mimari ekrana basılıyor...")
    num_steps = len(pipeline_results)
    
    # 2 satır ve 4 sütunluk (Toplam 8 yer) bir ızgara oluşturuyoruz
    plt.figure(figsize=(20, 10)) 
    
    for i, (title, img_data) in enumerate(pipeline_results.items()):
        # 2x3 yerine 2x4 yapıyoruz ki 7. resim sığsın
        plt.subplot(2, 4, i + 1) 
        plt.title(title, fontsize=12, fontweight='bold')
        plt.imshow(np.clip(img_data, 0.0, 1.0))
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    test_image = "C:/Users/bozku/Desktop/low_light/Genel_Dataset/Test/7a.jpg"
    model_weights = "snapshots_new3/Epoch200.pth"
    
    run_hybrid_pipeline(test_image, model_weights)