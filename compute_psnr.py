import os
import glob
import cv2
import math
import numpy as np
from tqdm import tqdm
import argparse
try:
    import imageio.v2 as imageio
except:
    import imageio

def read_img(img_path): # read image in rgb
    img = imageio.imread(img_path)
    # some images have 4 channels, such as DIV2KRK
    if img.shape[2] > 3: img = img[:, :, :3]
    return img

def calculate_psnr(img1, img2):
    ""
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    "only use to calculate ssim in y channel"
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

def calc_psnr_ssim(img1, img2, scale=2):
    assert scale != 0
    img1_crop = img1[scale:-scale, scale:-scale]
    img2_crop = img2[scale:-scale, scale:-scale]
    psnr = calculate_psnr(img1_crop, img2_crop)
    ssim = calculate_ssim(img1_crop, img2_crop)
    return psnr, ssim

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def test_one(args, in_path, gt_path):
    
    sr_img = read_img(in_path) / 255. # convert to [0,1], and aviod round() for int8
    sr_img_y = rgb2ycbcr(sr_img) * 255.
    
    hr_img = read_img(gt_path) / 255.
    hr_img_y = rgb2ycbcr(hr_img) * 255.

    psnr_, ssim_ = calc_psnr_ssim(sr_img_y, hr_img_y, args.scale)

    return psnr_, ssim_

def test_all(args, in_paths, gt_paths):
    psnrs, ssims = [], []
    for in_path, gt_path in tqdm(zip(in_paths, gt_paths), ncols=80):
        psnr_, ssim_ = test_one(args, in_path, gt_path)
        psnrs.append(psnr_)
        ssims.append(ssim_)

    return psnrs, ssims


def main(args):
    in_paths = sorted(glob.glob(os.path.join(args.input_folder, "*", "*.png")))
    gt_paths = sorted(glob.glob(os.path.join(args.gt_folder, "*.png")))
    for in_path, gt_path in zip(in_paths, gt_paths):
        in_name = os.path.basename(in_path).replace('ZSSR_', '')
        gt_name = os.path.basename(gt_path)
        assert in_name == gt_name

    psnrs, ssims = test_all(args, in_paths, gt_paths)
    print("PSNR/SSIM: {:.3f}/{:.4f} ".format(np.mean(psnrs), np.mean(ssims)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/mnt/cephfs/home/dengzeshuai/code/SuperResolution/KernelGAN/results_cubic_ZSSR',
                    help='path to test image')
    parser.add_argument('--gt_folder', type=str, default='/mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/gt_rename/',
                    help='path to gt image for metric computing')
    parser.add_argument('--scale', type=int, default=2,
                    help='the scale of SR, decide how many border pixels to be crop')
    args = parser.parse_args()
    print(args)
    main(args)