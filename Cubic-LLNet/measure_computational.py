import torch
import time
from thop import profile
# Senin modelin (LCE-Net veya Zero-DCE)
from LLE_NET import LLE_Net

def measure_computational_cost():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LLE_Net().to(device)
    model.eval()

    # Otonom araç kameraları genelde 640x480 veya 512x512 çözünürlük kullanır
    # Sisteme test için boş bir "Dummy" tensör (resim) gönderiyoruz
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # 1. FLOPs ve Parametre Hesaplama
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  # MACs (Multiply-Accumulate) genellikle 2 FLOPs'a eşittir
    
    print(f"--- MODEL MALİYET RAPORU ---")
    print(f"Toplam Parametre: {params / 1e6:.4f} Milyon ({(params):,} Parametre)")
    print(f"Hesaplama Maliyeti (GFLOPs): {flops / 1e9:.4f} GFLOPs")

    # 2. Hız (FPS / Inference Time) Ölçümü
    # GPU ısınması (warm-up) için 10 tur boş çalıştır
    for _ in range(10):
        _ = model(dummy_input)

    # Gerçek zaman ölçümü (100 karenin ortalaması)
    iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_time_ms

    print(f"Ortalama Çıkarım Süresi: {avg_time_ms:.2f} milisaniye (ms)")
    print(f"Saniyedeki Kare Hızı (FPS): {fps:.2f} FPS")

# Çalıştır
measure_computational_cost()