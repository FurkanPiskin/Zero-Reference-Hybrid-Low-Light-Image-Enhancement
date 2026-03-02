import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Giriş 2 kanal olacak: (AvgPool + MaxPool)
        # Çıkış 1 kanal olacak: (Dikkat Maskesi)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Kanal ekseninde (dim=1) Ortalama ve Maksimum al
        # Bu işlem 'Neresi parlak?', 'Neresi dolu?' bilgisini sıkıştırır.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # İkisini birleştir (Concatenate)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Konvolüsyon + Sigmoid ile 0-1 arası maske üret
        out = self.conv1(x_cat)
        mask = self.sigmoid(out)
        
        # Orijinal özellik haritasını bu maske ile çarp
        return x * mask