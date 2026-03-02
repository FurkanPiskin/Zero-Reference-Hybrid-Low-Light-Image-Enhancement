import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from LLE_NET import LLE_Net
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modeli yükleme ---
epoch_path = "snapshots_new3/Epoch200.pth"
model = LLE_Net().to(device)
checkpoint = torch.load(epoch_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Görüntüyü yükleme ---
#img_path = "Image_Process/Image_Enhacement_Dataset/LowDataset/"
img_path="C:/Users/bozku/Desktop/low_light/Genel_Dataset/Test/2015_05724.png"

orig = Image.open(img_path).convert("RGB")
orig_np = np.array(orig).astype(np.float32) / 255.0
to_tensor = transforms.ToTensor()
input_tensor = to_tensor(orig).unsqueeze(0).to(device)

# --- İyileştirme ---
with torch.no_grad():
    _, _, _, enh_final, _ = model(input_tensor)

# --- Tensor'u numpy'a çevirme ---
enh_final_np = enh_final.squeeze(0).cpu().permute(1,2,0).numpy()
enh_final_np = np.clip(enh_final_np, 0, 1)

#cv2.imwrite("enhanced_image2.png", (enh_final_np * 255).astype(np.uint8))
cv2.imwrite("enhanced_image2.png", cv2.cvtColor((enh_final_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


# --- Görselleştirme ---
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig_np)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Final Enhanced")
plt.imshow(enh_final_np)
plt.axis('off')

plt.show()
