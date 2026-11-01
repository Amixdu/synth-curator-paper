"""
Orchestrator for EDM2 batch generation.
Maps a task index (read from SLURM_ARRAY_TASK_ID) to a (checkpoint, CFG scale) pair.
Generates exact, class-specific volumes based on the config.
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
    
    # Read SLURM Array ID
    try:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
        if task_id == -1: raise ValueError
        print(f"--- Processing Task ID: {task_id} ---")
    except (TypeError, ValueError):
        print("ERROR: SLURM_ARRAY_TASK_ID not found.")
        sys.exit(1)

    # Build Job Grid
    checkpoints = gen_params['parameter_space']['checkpoints']
    cfg_scales = gen_params['parameter_space']['cfg_scales']
    
    job_list = []
    for ckpt_name, ckpt_file in checkpoints.items():
        for cfg in cfg_scales:
            job_list.append({
                'ckpt_name': ckpt_name,
                'ckpt_file': ckpt_file,
                'cfg': cfg,
            })

    if task_id >= len(job_list):
        print(f"Task ID {task_id} out of bounds.")
        sys.exit(0)
        
    job = job_list[task_id]
    ckpt_path = Path(gen_params['checkpoint_dir']) / job['ckpt_file']
    cfg_val = job['cfg']
    
    # Setup Output
    # Structure: .../synthetic_pool/edm2/ckpt_3k_cfg_1.5/class_0/
    base_out = Path(gen_params['output_root_dir']) / f"ckpt_{job['ckpt_name']}_cfg_{cfg_val}"
    base_out.mkdir(parents=True, exist_ok=True)
    gen_script = gen_section['scripts']['generation_script']
    
    # Load Class Distributions
    class_distributions = gen_params['settings']['class_distributions']
    gen_batch_size = gen_params['settings'].get('batch_size', 32)
    
    print(f"Generating for: {job['ckpt_name']} @ CFG {cfg_val}")
    
    current_seed_offset = 0
    for class_idx_str, required_count in class_distributions.items():
        class_idx = int(class_idx_str)
        
        start_seed = current_seed_offset
        end_seed = start_seed + required_count - 1

        current_seed_offset += required_count
        
        class_out = base_out / f"class_{class_idx}"
        class_out.mkdir(parents=True, exist_ok=True)
        
        # Idempotency Check
        existing = len(list(class_out.glob('*.png')))
        if existing >= required_count:
            print(f"  [SKIP] Class {class_idx} done ({existing} images, target {required_count}).")
            continue
            
        print(f"  [RUN] Class {class_idx} (Target: {required_count} | Seeds {start_seed}-{end_seed})")
        
        cmd = [
            'torchrun', '--standalone', '--nproc_per_node=1', 
            gen_script,
            f"--net={ckpt_path}",
            f"--outdir={class_out}",
            f"--seeds={start_seed}-{end_seed}",
            f"--class={class_idx}",
            f"--guidance={cfg_val}",
            f"--batch={gen_batch_size}",
            f"--steps=40" # Standard EDM2 steps
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Generation failed for class {class_idx}")
            sys.exit(1)

    print("Task Complete.")

if __name__ == "__main__":
    main()