"""
Padding orchestrator for EDM2.

Pads the NV class (class 5) from seed 1786 up to seed 6500, adding about 4,715 NV images
per checkpoint x CFG configuration. NV makes up about two thirds of the target class
distribution, so a curated set drawn from a single configuration needs more than 5,000 NV
images, while the initial pass produced only 1,786 per configuration. This script fills
that gap.
"""
import os
import subprocess
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2] 
sys.path.append(str(project_root))
from utils.helpers import load_and_resolve_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_and_resolve_config(Path(args.config))
    gen_section = config['image_generation']['edm']
    gen_params = gen_section['generation_params']
    
    try:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
        if task_id == -1: raise ValueError
        print(f"--- Processing Task ID: {task_id} (NV Padding) ---")
    except (TypeError, ValueError):
        print("ERROR: SLURM_ARRAY_TASK_ID not found.")
        sys.exit(1)

    checkpoints = gen_params['parameter_space']['checkpoints']
    cfg_scales = gen_params['parameter_space']['cfg_scales']
    
    job_list = []
    for ckpt_name, ckpt_file in checkpoints.items():
        for cfg in cfg_scales:
            job_list.append({'ckpt_name': ckpt_name, 'ckpt_file': ckpt_file, 'cfg': cfg})

    if task_id >= len(job_list):
        sys.exit(0)
        
    job = job_list[task_id]
    ckpt_path = Path(gen_params['checkpoint_dir']) / job['ckpt_file']
    cfg_val = job['cfg']
    
    base_out = Path(gen_params['output_root_dir']) / f"ckpt_{job['ckpt_name']}_cfg_{cfg_val}"
    base_out.mkdir(parents=True, exist_ok=True)
    gen_script = gen_section['scripts']['generation_script']
    gen_batch_size = gen_params['settings'].get('batch_size', 32)
    
    print(f"Generating Padding for: {job['ckpt_name']} @ CFG {cfg_val}")
    
    # =================================================================
    # THE PADDING LOGIC (Only Class 5, Seeds 1786 to 6500)
    # =================================================================
    class_idx = 5  # NV is Class 5
    start_seed = 1786
    end_seed = 6500  # Gives us ~4715 new NV images per config (Total NV = 6501)
    
    class_out = base_out / f"class_{class_idx}"
    class_out.mkdir(parents=True, exist_ok=True)
    
    # Check if we already padded this
    existing = len(list(class_out.glob('*.png')))
    if existing >= 6000:
        print(f"  [SKIP] NV Class 5 already padded ({existing} images).")
        sys.exit(0)
        
    print(f"  [RUN] Padding NV Class 5 (Seeds {start_seed}-{end_seed})")
    
    cmd = [
        'torchrun', '--standalone', '--nproc_per_node=1',
        gen_script,
        f"--net={ckpt_path}",
        f"--outdir={class_out}",
        f"--seeds={start_seed}-{end_seed}",
        f"--class={class_idx}",
        f"--guidance={cfg_val}",
        f"--batch={gen_batch_size}",
        f"--steps=40" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Generation failed for padding.")
        sys.exit(1)

    print("Task Complete.")

if __name__ == "__main__":
    main()