# -*- coding: utf-8 -*-
from PIL import Image
import os
from tqdm import tqdm

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()

    dir_origin_path = "img/"
    dir_save_path   = "proposal/"
    
    #显示proposal的数量
    count=8

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image     = frcnn.show_proposals(image,count)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)


