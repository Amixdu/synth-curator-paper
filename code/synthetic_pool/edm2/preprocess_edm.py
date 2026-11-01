"""
EDM2 Data Preprocessor & VAE Safety Auditor.
Scope: Training Preparation Only.
1. Staging
2. VAE Safety Audit
3. RGB Zip (Intermediate step for encoding)
4. Latent Zip (Final Training Input)
"""
import sys
import argparse
import os
import shutil
import random
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKL

# Add project root to path
project_root = Path(__file__).resolve().parents[2] 
sys.path.append(str(project_root))

from utils.helpers import load_and_resolve_config

# -----------------------------------------------------------------------------
# SSIM Utility
# -----------------------------------------------------------------------------
def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type(img1.type())
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
def run_edm_script(script_name, args_list, working_dir):
    """Runs an internal EDM2 python script."""
    # We use script_name (string) not script_path (Path object) to be safe
    cmd = [sys.executable, script_name] + args_list
    print(f"\n[EXEC] Running: {' '.join(cmd)}")
    print(f"       In Dir:  {working_dir}")
    subprocess.run(cmd, check=True, cwd=working_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_and_resolve_config(Path(args.config))

    # --- Config Extraction ---
    source_real = Path(config['data']['source_real']['split']['train'])
    temp_dir = Path(config['image_generation']['edm']['preprocessing']['train_data_source_dir'])
    
    rgb_zip = Path(config['image_generation']['edm']['paths']['dataset_rgb_zip'])
    latent_zip = Path(config['image_generation']['edm']['paths']['dataset_latent_zip'])
    vae_audit_dir = Path(config['image_generation']['edm']['paths']['vae_audit_dir'])
    
    # We extract 'dir' and 'file' separately to ensure we point to the correct folder
    tool_conf = config['scripts']['image_generation']['edm']['preprocessing']['dataset_tool_script']
    tool_dir = Path(tool_conf['dir'])
    tool_script_name = tool_conf['file']
    
    res_str = str(config['image_generation']['edm']['preprocessing']['resolution'])
    resolution = int(res_str.split('x')[0]) if 'x' in res_str else int(res_str)

    audit_samples_n = config['image_generation']['edm']['preprocessing'].get('vae_audit_samples', 50)
    ssim_thresh = config['image_generation']['edm']['preprocessing'].get('ssim_threshold', 0.80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Preprocessing on: {device}")

    # --- Step 1: Staging ---
    print(f"\n[Step 1] Staging data from {source_real}...")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    shutil.copytree(source_real, temp_dir)
    
    # Robust globbing
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        all_images.extend(list(temp_dir.glob(ext)))
        all_images.extend(list(temp_dir.glob(f"**/{ext}")))
    
    # Deduplicate
    all_images = list(set(all_images))
    
    print(f"Found {len(all_images)} images in staging area.")
    if len(all_images) == 0:
        print("CRITICAL ERROR: No images found. Check source path.")
        sys.exit(1)

    # --- Step 2: VAE Safety Audit ---
    print(f"\n[Step 2] VAE Safety Audit...")
    if vae_audit_dir.exists(): shutil.rmtree(vae_audit_dir)
    vae_audit_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAE...")
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.eval()
        vae.to(device)
    except Exception as e:
        print(f"VAE Load Error: {e}")
        sys.exit(1)

    # Sample some images
    sample_count = min(audit_samples_n, len(all_images))
    samples = random.sample(all_images, sample_count)
    print(f"Auditing {len(samples)} samples...")
    
    scores = []
    with torch.no_grad():
        for i, img_path in enumerate(samples):
            # Load & Resize
            img = Image.open(img_path).convert('RGB')
            img = img.resize((resolution, resolution), Image.LANCZOS)
            
            # To Tensor
            img_t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)

            # VAE Roundtrip
            x = (img_t * 2.0) - 1.0
            latents = vae.encode(x).latent_dist.sample() * 0.18215
            latents = latents / 0.18215
            recon = vae.decode(latents).sample
            recon = (recon / 2 + 0.5).clamp(0, 1)

            # Metrics
            s_val = ssim(img_t, recon).item()
            scores.append(s_val)

            # Save Side-by-Side
            recon_img = Image.fromarray((recon[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            combined = Image.new('RGB', (resolution * 2, resolution))
            combined.paste(img, (0, 0))
            combined.paste(recon_img, (resolution, 0))
            
            clean_name = img_path.stem
            save_path = vae_audit_dir / f"audit_{i}_{clean_name}_ssim_{s_val:.2f}.png"
            combined.save(save_path)
            print(f"  [{i+1}/{sample_count}] {clean_name}: SSIM={s_val:.4f}")

    avg_ssim = sum(scores) / len(scores)
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Adjusted threshold based on your previous run
    if avg_ssim < ssim_thresh: 
        print(f"WARNING: VAE fidelity ({avg_ssim:.4f}) is below threshold ({ssim_thresh}).")
        print("Check 'vae_safety_check' folder for artifact analysis.")
    else:
        print("VAE fidelity passed.")

    del vae
    torch.cuda.empty_cache()

    # --- Step 3: Create RGB Zip ---
    print(f"\n[Step 3] Creating RGB Zip (Source for Latents)...")
    if rgb_zip.exists(): rgb_zip.unlink()
    rgb_zip.parent.mkdir(parents=True, exist_ok=True)
    
    run_edm_script(
        tool_script_name,
        ['convert', '--source', str(temp_dir), '--dest', str(rgb_zip), '--resolution', f"{resolution}x{resolution}"],
        tool_dir
    )

    # --- Step 4: Create Latent Zip ---
    print(f"\n[Step 4] Creating Latent Zip (Training Source)...")
    if latent_zip.exists(): latent_zip.unlink()
    
    run_edm_script(
        tool_script_name,
        ['encode', '--source', str(rgb_zip), '--dest', str(latent_zip)],
        tool_dir
    )

    print(f"\n[SUCCESS] Preprocessing Complete.")
    print(f"Latents Ready: {latent_zip}")

if __name__ == "__main__":
    main()