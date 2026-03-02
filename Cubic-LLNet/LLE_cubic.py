import torch
import torch.nn.functional as F
class LLE_cubic():
    """
    @staticmethod
    def LLE_LUT(origin_img, t, b):
                batch_size, c, h, w = origin_img.shape
                # 对t, b升维（batch_size, 3, 256, 1）
                #t = t.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()
                #b = b.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()

                x_init = (torch.arange(256).reshape(-1, 1) / 255.0).requires_grad_(True).cuda()
                x_init = torch.unsqueeze(torch.unsqueeze(x_init, 0), 0).expand(batch_size, 3, 256, 1)

                # 根据变换规则进行映射
                lookup_tables = ((x_init - t) ** 3 + t ** 3) * (1 - b) / ((1 - t) ** 3 + t ** 3) + b * x_init
                # 变为 0~255之间的像素值
                lookup_tables = torch.round(lookup_tables * 255)

                origin_img = origin_img * 255
                #origin_img = origin_img.view(batch_size, c, -1, 1).to(dtype=torch.int64)  # 将输入向量变为（b, c, h*w, 1）
                origin_img = origin_img.view(batch_size, c, -1, 1).to(dtype=torch.int64).cuda()

                enhance_image = torch.gather(lookup_tables, dim=2, index=origin_img)
                enhance_image = enhance_image / 255.0
                enhance_image = enhance_image.view(batch_size, c, h, w)

                return enhance_image
        """
        
    @staticmethod
    def LLE_LUT(origin_img, t, b):
        """
        origin_img: (Batch, 3, H, W) -> Giriş Görüntüsü
        t: (Batch, 3, H, W) -> Curve Parametresi 1 (Harita)
        b: (Batch, 3, H, W) -> Curve Parametresi 2 (Harita)
        """

        # 1. Boyut Güvenlik Kontrolü
        # Eğer modelden çıkan t ve b haritası, orijinal resimden küçükse (padding yüzünden vs.)
        # onları resim boyutuna "uzat" (Interpolate).
        if t.shape[-2:] != origin_img.shape[-2:]:
            t = F.interpolate(t, size=origin_img.shape[-2:], mode='bilinear', align_corners=True)
            b = F.interpolate(b, size=origin_img.shape[-2:], mode='bilinear', align_corners=True)

        # 2. Formülün Uygulanması (Pixel-wise)
        # Senin kodundaki formül: ((x - t)^3 + t^3) * (1 - b) / ((1 - t)^3 + t^3) + b * x
        # Burada 'x' artık 'origin_img'in kendisi. LUT kullanmıyoruz, direkt hesaplıyoruz.
        
        # Pay (Numerator)
        numerator = (torch.pow(origin_img - t, 3) + torch.pow(t, 3)) * (1 - b)
        
        # Payda (Denominator) + 1e-8 (Sıfıra bölünme hatasını önlemek için küçük sayı)
        denominator = (torch.pow(1 - t, 3) + torch.pow(t, 3)) + 1e-8
        
        # Ana Formül
        enhance_image = (numerator / denominator) + b * origin_img

        # 3. Sonuç Kontrolü (0-1 aralığında tutmak için)
        # enhance_image = torch.clamp(enhance_image, 0, 1) # İsteğe bağlı, Sigmoid zaten 0-1 veriyor ama garanti olsun.

        return enhance_image
    """
        Piksel bazlı kübik (cubic) dönüşüm ile giriş görüntüsünü iyileştirir.
        
        Parametreler:
            origin_img (Tensor): Giriş görüntü tensörü, şekil (batch_size, kanallar, yükseklik, genişlik), değerler [0,1] aralığında.
            t (Tensor): Kübik dönüşüm için öğrenilen eşik parametresi, şekil (batch_size, 3).
            b (Tensor): Kübik dönüşüm için öğrenilen karıştırma parametresi, şekil (batch_size, 3).
        
        Döndürür:
            enhance_image (Tensor): Aynı boyutlarda iyileştirilmiş görüntü tensörü, değerler [0,1] aralığında.
        
        Açıklama:
            - t ve b parametreleri piksel boyutlarına uygun şekilde genişletilir (broadcast edilir).
            - Her piksel için ayrı ayrı kübik dönüşüm uygulanır.
            - Oluşturulan lookup table, orijinal piksel değerlerini iyileştirilmiş değerlerle eşler.
    """
    @staticmethod
    def generate_lookup_table(batch_size, t, b):
                '''
                本函数为生成对应查找表的函数。
                :param batch_size:pytorch 中 batch_size 值。
                :param t: 维度为（batch_size, 3）,其中3 为对应的3通道
                :param b: 维度为（batch_size, 3）,其中3 为对应的3通道
                :return: 依据规则生成的查找表， 维度信息为（batch_size, 3, 256)
                '''

                # print("-------------t is---------")
                # print(t)
                # print("-------------b is---------")
                # print(b)

                # 对t, b升维（batch_size, 3, 256, 1）
                t = t.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()
                b = b.unsqueeze(2).unsqueeze(2).expand(batch_size, 3, 256, 1).requires_grad_(True).cuda()

                x_init = (torch.arange(256).reshape(-1, 1) / 255.0).requires_grad_(True).cuda()
                x_init = torch.unsqueeze(torch.unsqueeze(x_init, 0), 0).expand(batch_size, 3, 256, 1)

                # 根据变换规则进行映射
                lookup_tables = ((x_init - t) ** 3 + t ** 3) * (1 - b) / ((1 - t) ** 3 + t ** 3) + b * x_init #Makaledeki formül
                # 变为 0~255之间的像素值
                lookup_tables = torch.round(lookup_tables * 255)
                lookup_tables = lookup_tables

                # print("==================lookup_table=================")
                # print(lookup_tables)
                # print("lookup_tables.shape", lookup_tables.shape)

                return lookup_tables
    
    @staticmethod
    def LLE_LUT_formula(images, lookup_table):
                '''
                本函数为将输入的图像输出为映射后的图像
                :param images: 输入的待映射图像，四维向量（batch_size, c, h, w）
                :param lookup_table: 查找表映射，（batch_size, 3, 256, 1）
                :return: 依据查找表的映射后图像
                '''

                # 由于传进来的都是归一化图像，因此再变换到0~255区间，维度为（batch_size, h, w, c）
                b, c, h, w = images.shape
                images_mem = images * 255.0
                images_mem = images_mem.view(b, c, -1, 1).to(dtype=torch.int64)  # 将输入向量变为（b, c, h*w, 1）
                imgdst = torch.gather(lookup_table, dim=2, index=images_mem)
                enhance_image = imgdst / 255.0
                enhance_image = enhance_image.view(b, c, h, w)

                # print("===========original_image is ==============  ")
                # print(images * 255)
                # print("enhance_image shape is ", enhance_image.shape)
                # print("===========enhance_image is ==============  ")
                # print(enhance_image * 255)
                # print("enhance_image shape is ", enhance_image.shape)

                return enhance_image
    
    def gamma_trans(x_origin, t, b):
            alpha_multi_gamma = 1/(torch.pow(t[:, :, None, None], b[:, :, None, None]))
            x_enhance = alpha_multi_gamma * ((t[:, :, None, None] * x_origin) ** b[:, :, None, None])

            return x_enhance
    
    def gamma_trans_simply(x_origin, t, b):
            x_enhance = torch.pow(x_origin, t[:, :, None, None]) + b[:, :, None, None]

            return x_enhance

    def gamma_simply(x_origin, t):

            x_enhance = torch.pow(x_origin, t[:, :, None, None])

            return x_enhance
    
