import os
import cv2 as cv
import argparse
import torch
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/BCI', help='Results saved path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    opt = parser.parse_args()
    return opt


opt = parse_opt()

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results.
    torch.backends.cudnn.benchmark = False  # Disables optimization for specific hardware.
    print(f"Random seed set to {seed}")

set_random_seed(opt.seed)

def evaluate_metrics(result_path):
    images_dir = os.path.join(result_path, 'test_latest', 'images')
    psnr_list = []
    ssim_list = []
    real_paths = []
    fake_paths = []

    # Process images
    for filename in tqdm(os.listdir(images_dir), desc="Processing images"):
        if 'fake' in filename:
            fake_path = os.path.join(images_dir, filename)
            real_filename = filename.replace('fake', 'real')
            real_path = os.path.join(images_dir, real_filename)
            if not os.path.exists(real_path):
                continue
            fake_paths.append(fake_path)
            real_paths.append(real_path)

            # Read images
            fake_img = cv.imread(fake_path)
            real_img = cv.imread(real_path)
            if fake_img is None or real_img is None:
                continue

            # Compute PSNR and SSIM
            psnr_score = peak_signal_noise_ratio(real_img, fake_img)
            ssim_score = structural_similarity(real_img, fake_img, channel_axis=2)
            psnr_list.append(psnr_score)
            ssim_list.append(ssim_score)

    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0

    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Compute FID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        num_avail_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_avail_cpus = os.cpu_count() or 1
    num_workers = min(num_avail_cpus, 8)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    print("\nComputing activation statistics for real images ...")
    m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                             device=device, num_workers=num_workers)
    print("Computing activation statistics for fake images ...")
    m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                             device=device, num_workers=num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"\nFID Score: {fid_value:.2f}")
if __name__ == '__main__':
    evaluate_metrics(opt.result_path)
