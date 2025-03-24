# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import os
import fnmatch


def jigsaw(dirPath, num):
    w_num = 10
    h_num = (num//w_num) if num%w_num==0 else (num//w_num+1)
    h = 64
    w = 64
    recon_imgs = []
    ref_imgs = []

    f_name_list = os.listdir(dirPath)

    for i in range(num):
        img1Path = fnmatch.filter(f_name_list, 'recon_{}_*'.format(i))[0]
        img2Path = fnmatch.filter(f_name_list, 'ref_{}_*'.format(i))[0]

        recon_imgs.append(Image.fromarray(cv2.imread(dirPath + img1Path)))
        ref_imgs.append(Image.fromarray(cv2.imread((dirPath + img2Path))))

    result = Image.new(recon_imgs[0].mode, (w_num*w, h_num*h*2))
    for i in range(h_num):
        for j in range(w_num):
            if i * w_num + j >= num:
                break
            result.paste(recon_imgs[i*w_num+j], box=(j*w, i*2*h))
            result.paste(ref_imgs[i*w_num+j], box=(j*w, (i*2+1)*h))

    return np.array(result)

if __name__ == '__main__':
    dir = ['front_layer_86', 'front_layer_90']
    img_num = 40
    for i in range(len(dir)):
        dirPath = './recon_pics/' + dir[i] + '/'
        print(dirPath)

        img = jigsaw(dirPath, img_num)
        # cv2.imwrite(dirPath + 'result_inv.png', img)
        cv2.imwrite(dirPath + 'result.png', img)