import cv2
import numpy as np

def pixel_detective_live(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Resim bulunamadı.")
        return

    print("--- PİKSEL DEDEKTİFİ BAŞLATILDI ---")
    print("1. Açılan pencerede resmin herhangi bir noktasına SOL TIKLA.")
    print("2. Konsoldaki değerleri oku.")
    print("3. Çıkmak için 'q' tuşuna bas.")
    print("-------------------------------------")

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # OpenCV renkleri BGR okur
            b = int(img[y, x, 0])
            g = int(img[y, x, 1])
            r = int(img[y, x, 2])
            
            print(f"\nKoordinat: ({x}, {y})")
            print(f"-> R: {r}, G: {g}, B: {b}")
            
            # Analiz Kısmı
            diffs = [abs(r-g), abs(g-b), abs(r-b)]
            max_diff = max(diffs)
            
            if max_diff < 3:
                print("⚠️ SONUÇ: Bu piksel NEREDEYSE GRİ (Renk bilgisi çok zayıf veya yok).")
            else:
                # Hangi renk baskın?
                if r > g and r > b:
                    print(f"✅ SONUÇ: Gizli Renk KIRMIZI! (Fark: {r - max(g,b)})")
                elif g > r and g > b:
                    print(f"✅ SONUÇ: Gizli Renk YEŞİL! (Fark: {g - max(r,b)})")
                elif b > r and b > g:
                    print(f"✅ SONUÇ: Gizli Renk MAVİ! (Fark: {b - max(r,g)})")
                else:
                    print("✅ SONUÇ: Karışık bir renk var (Gri değil).")

    cv2.imshow('Piksel Dedektifi (Cikmak icin q)', img)
    cv2.setMouseCallback('Piksel Dedektifi (Cikmak icin q)', click_event)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# O sisli kedi veya şişe resmini buraya koy
pixel_detective_live('enhanced_image2.png')