import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)

# folder_path = "sum"  # 文件夹路径
#
# # 遍历文件夹及其子文件夹中的所有文件
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         file_path = os.path.join(root, file)
#         print(file_path)


def LOL_train_dataset(lowlight_images_path):
    file_paths_and_names = []

    # 遍历文件夹
    for dirpath, dirnames, filenames in os.walk(lowlight_images_path):
        # 遍历文件
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths_and_names.append(file_path)

    random.shuffle(file_paths_and_names)

    return file_paths_and_names


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.png")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = LOL_train_dataset(lowlight_images_path)
        self.size = 256
        print("Total dataset examples:", len(self.train_list))
        if len(self.train_list) > 1000:
            self.train_list = random.sample(self.train_list, 30195)

        self.data_list = self.train_list
        print("Total training examples:", len(self.data_list))



    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        data_lowlight = data_lowlight.convert("RGB")
        
        #data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS) Eski sürümlerde
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.Resampling.LANCZOS)

        
        data_lowlight = (np.asarray(data_lowlight) / 255.0)

     
        data_lowlight = torch.from_numpy(data_lowlight).float()

      
        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    lowlight_images_path = "Image_Process/Image_Enhacement_Dataset/LowDataset/"
    
    files = LOL_train_dataset(lowlight_images_path)
    print("共有图像{}张".format(len(files)))

    
    # for file in files:
    #     print(file)
    low_light_dataset = lowlight_loader(lowlight_images_path)





