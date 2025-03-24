from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import os
import fnmatch
import torch
from piq import psnr, ssim, LPIPS
import torch.nn.functional as F

from niqe import niqe


def count_PSNR(img1, img2):
    return psnr(img1, img2, data_range=255).item()

def count_SSIM(img1, img2):
    return ssim(img1, img2, data_range=255).item()

def count_LPIPS(img1, img2):
    img1 = torch.tensor(img1, dtype=torch.float32) / 255
    img2 = torch.tensor(img2, dtype=torch.float32) / 255

    lpips = LPIPS()
    return lpips(img1, img2).item()

if __name__ == '__main__':
    dir = ['front_layer_90']
    img_num = 40
    for i in range(len(dir)):
        dirPath = './recon_pics/' + dir[i] + '/'
        print(dirPath)

        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_niqe = 0
        total_mse = 0

        note_psnr = open(dirPath + 'psnr.txt', 'w')
        note_ssim = open(dirPath + 'ssmi.txt', 'w')
        note_lpips = open(dirPath + 'lpips.txt', 'w')
        note_niqe = open(dirPath + 'niqe.txt', 'w')
        note_mse = open(dirPath + 'mse.txt', 'w')

        f_name_list = os.listdir(dirPath)

        for ii in range(img_num):
            img1Path = fnmatch.filter(f_name_list, 'recon_{}_*'.format(ii))
            img2Path = fnmatch.filter(f_name_list, 'ref_{}_*'.format(ii))
            img1 = torch.tensor(np.array(Image.open(dirPath + img1Path[0]))).permute(2,0,1).unsqueeze(0)
            img2 = torch.tensor(np.array(Image.open(dirPath + img2Path[0]))).permute(2,0,1).unsqueeze(0)

            psnr_val = count_PSNR(img1, img2)
            note_psnr.write('[' + str(ii) + '] PSNR=' + str(psnr_val) + '\n')
            total_psnr += psnr_val

            ssim_val = count_SSIM(img1, img2)
            note_ssim.write('[' + str(ii) + '] SSIM=' + str(ssim_val) + '\n')
            total_ssim += ssim_val

            lpips_val = count_LPIPS(img1, img2)
            note_lpips.write('[' + str(ii) + '] LPIPS=' + str(lpips_val) + '\n')
            total_lpips += lpips_val

            niqe_val = niqe(dirPath + img1Path[0])
            note_niqe.write('[' + str(ii) + '] NIQE=' + str(niqe_val) + '\n')
            total_niqe += niqe_val

            mse_val = F.mse_loss(img1.float(), img2.float()).item()
            note_mse.write('[' + str(ii) + '] MSE=' + str(mse_val) + '\n')
            total_mse += mse_val

        avg_psnr = total_psnr / img_num
        note_psnr.write('avg PSNR=' + str(avg_psnr) + '\n')
        note_psnr.close()

        avg_ssim = total_ssim / img_num
        note_ssim.write('avg SSIM=' + str(avg_ssim) + '\n')
        note_ssim.close()

        avg_lpips = total_lpips / img_num
        note_lpips.write('avg LPIPS=' + str(avg_lpips) + '\n')
        note_lpips.close()

        avg_niqe = total_niqe / img_num
        note_niqe.write('avg NIQE=' + str(avg_niqe) + '\n')
        note_niqe.close()

        avg_mse = total_mse / img_num
        note_mse.write('avg MSE=' + str(avg_mse) + '\n')
        note_mse.close()