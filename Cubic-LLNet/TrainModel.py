import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import time
import numpy as np
import DataLoader as dataloader
from LLE_NET import LLE_Net
import Loss
from tqdm import tqdm  # DÜZELTME: Modül hatasını önlemek için 'from' ile import

def weights_init(m):
    """Ağ katmanlarını normal dağılımla başlatır"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(
    lowlight_images_path="Image_Process/Image_Enhacement_Dataset/LowDataset",
    lr=0.00005,
    weight_decay=0.0001,
    grad_clip_norm=0.1,
    num_epochs=50,
    train_batch_size=16, # GÜNCELLEME: 8 -> 16 yapıldı
    num_workers=4,
    snapshots_folder="snapshots/",
    resume=False,
    resume_path="snapshots/Epoch53.pth"
):
    """Eğitim fonksiyonu"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === MODEL ===
    DCE_net =LLE_Net().to(device)
    
    # Optimizer (Resume durumunda state_dict yüklenmesi için önce tanımlanır)
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    start_epoch = 0

    # === EĞİTİME DEVAM ===
    if resume and os.path.exists(resume_path):
        print(f"📂 Önceki eğitimden devam ediliyor: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        DCE_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # --- KRİTİK DÜZELTME BAŞLANGICI ---
        # State yüklendikten sonra, parametre olarak gelen yeni LR'yi zorla uygula
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"🔧 Learning Rate güncellendi: {lr}")
        # --- KRİTİK DÜZELTME BİTİŞİ ---

        start_epoch = checkpoint["epoch"] + 1
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("   -> Scheduler durumu yüklendi.")
        
        print(f"   -> Kaldığı yerden devam: Epoch {start_epoch}")

    # === SCHEDULER (DÜZELTME) ===
    # if-else bloğunun dışına alındı, böylece resume=True olsa da hata vermez.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Eğer resume yapıldıysa scheduler'ı da doğru epoch'a senkronize etmek gerekebilir.
    # Ancak StepLR her step() çağrısında güncellendiği için döngü içinde kendini toparlar.

    cudnn.benchmark = True
    DCE_net.train()

    # === LOSS FONKSİYONLARI ===
    L_Angle_Loss = Loss.ColorAngleLoss().to(device)
    L_color = Loss.L_color().to(device)
    L_spa = Loss.L_spa().to(device)
    L_exp = Loss.L_exp(16, 0.7).to(device)
    L_contrast = Loss.L_ContrastLoss(4).to(device)

    # === VERİ YÜKLE ===
    train_dataset = dataloader.lowlight_loader(lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if not os.path.exists(snapshots_folder):
        os.makedirs(snapshots_folder)

    # === EĞİTİM DÖNGÜSÜ ===
    print(f"🚀 Eğitim Başlıyor: {start_epoch} -> {num_epochs} Epoch (Batch Size: {train_batch_size})")
    
    for epoch in range(start_epoch, num_epochs):
        # Kayıp değerlerini ayrı ayrı takip etmek için listeler
        epoch_loss_spa = []
        epoch_loss_angle = []
        epoch_loss_exp = []
        epoch_loss_con = []
        epoch_total_loss = []
        
        # TQDM ile İlerleme Çubuğu
        loop = tqdm(train_loader, leave=True)
        
        for i,img_lowlight in enumerate(loop):
            img_lowlight = img_lowlight.to(device)

            # İleri yayılım
            enh1, enh2, enh3, enh_final, _ = DCE_net(img_lowlight)

            # Loss Ağırlıkları
            weights = [150, 50, 3, 10] #40 15 5 8,80,40,2,8
            
            # Ayrı ayrı loss hesapla
            loss_spa_val= weights[0] * torch.mean(L_spa(enh_final, img_lowlight))
            loss_angle_val = weights[1] * torch.mean(L_Angle_Loss(enh_final, img_lowlight))
            #loss_col_val = weights[2] * torch.mean(L_color(enh_final))
            loss_exp_val = weights[2] * torch.mean(L_exp(enh_final))
            loss_con_val = weights[3] * torch.mean(L_contrast(enh_final))
            
            # Toplam Loss
            loss = loss_spa_val + loss_angle_val + loss_exp_val + loss_con_val

            # Listelere ekle
            epoch_loss_spa.append(loss_spa_val.item())
            epoch_loss_angle.append(loss_angle_val.item())
            epoch_loss_exp.append(loss_exp_val.item())
            epoch_loss_con.append(loss_con_val.item())
            epoch_total_loss.append(loss.item())

            # Geri yayılım
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), grad_clip_norm)
            optimizer.step()

            if (epoch == 20) and (i % 100 == 0): # İlk epochta, her 100 adımda bir
                for name, param in DCE_net.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        print(f"🔍 DEBUG: {name} katmanı gradyanı: {grad_mean:.8f}")
                        if grad_mean > 0:
                            print("✅ Gradyan akışı BAŞARILI!")
                            break
                    else:
                        print(f"❌ UYARI: {name} katmanında gradyan YOK!")

            # Barı güncelle
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        # Epoch sonunda Scheduler güncellemesi
        scheduler.step()
        
        # === Epoch Sonu Raporlama (AYRINTILI) ===
        avg_total = np.mean(epoch_total_loss)
        avg_spa = np.mean(epoch_loss_spa)
        avg_angle = np.mean(epoch_loss_angle)
        avg_exp = np.mean(epoch_loss_exp)
        avg_con = np.mean(epoch_loss_con)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] Tamamlandı | LR: {current_lr:.6f}")
        print(f"   🔴 Total Loss: {avg_total:.4f}")
        print(f"   DETAILS -> Spatial: {avg_spa:.4f} | Color: {avg_angle:.4f} | Exposure: {avg_exp:.4f} | Contrast: {avg_con:.4f}")

        # === Her epoch sonunda kaydet ===
        snapshot_path = os.path.join(snapshots_folder, f"Epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": DCE_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() 
        }, snapshot_path)

    print("🎯 Eğitim başarıyla tamamlandı.")

if __name__ == "__main__":
    start = time.time()

    train(
        lowlight_images_path="Genel_Dataset/Train", # Veri seti yolunu kontrol et
        lr=0.000001,
        weight_decay=0.0001,
        grad_clip_norm=0.1,
        num_epochs=200,
        train_batch_size=16, # Batch size 16 olarak ayarlandı
        num_workers=2,
        snapshots_folder="snapshots_new3/",
        resume=True,
        resume_path="snapshots_new3/Epoch195.pth"
    )

    end = time.time()
    print(f"================= Toplam Süre: {end - start:.2f} saniye =================")