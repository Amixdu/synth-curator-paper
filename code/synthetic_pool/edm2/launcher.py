"""
Launcher for EDM2 (Elucidated Diffusion Model v2) Training from a Central Configuration File.

This script acts as the primary entry point for initiating EDM2 training runs,
It reads a YAML configuration
file, dynamically constructs the full `torchrun` command with the appropriate
arguments, and executes it.

This approach ensures reproducibility, scalability, and a clean separation between
the experimental configuration and the execution environment.

It handles specific logic required for Latent Diffusion, including mapping configuration
units (kimg/ticks) to the raw image counts required by the EDM2 training engine,
resolving architecture presets, and enforcing safety hyperparameters (e.g., dropout).

Usage:
    $ python launcher.py --config /path/to/config_edm2.yml
"""
import subprocess
import sys
import argparse
import os
from pathlib import Path

# Add project root to Python path to allow importing utility functions
project_root = Path(__file__).resolve().parents[2] 
sys.path.append(str(project_root))

from utils.helpers import load_and_resolve_config

def main():
    parser = argparse.ArgumentParser(description="Launcher for EDM2 training.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    args = parser.parse_args()

    config = load_and_resolve_config(Path(args.config))
    
    # --- Extract Configuration Blocks ---
    edm_config = config['image_generation']['edm']
    paths = edm_config['paths']
    params = edm_config['training_params']
    preprocessing = edm_config['preprocessing']
    
    train_script = Path(__file__).parent / 'train_edm2.py'
    
    if not train_script.exists():
        print(f"[CRITICAL ERROR] Could not find training script at {train_script}")
        sys.exit(1)

    # --- Logic 1: Architecture Preset ---
    res_str = str(preprocessing['resolution'])
    resolution = res_str.split('x')[0] if 'x' in res_str else res_str
    
    # STRICT: Config must specify model_size (e.g., 's')
    model_size = params['model_size']
    preset_name = f"edm2-img{resolution}-{model_size}"

    # --- Logic 2: Unit Conversion ---
    # EDM2 operates on 'nimg' (raw image count) and requires binary alignment (multiples of 1024).
    # We define 1 kimg as 1024 images (Binary 'Ki'), not 1000, to satisfy internal assertions.
    MULTIPLIER = 1024 

    print("--- Unit Conversion (Config -> EDM2) ---")
    
    # 1. Tick Length
    tick_kimg = params['tick_duration']
    status_nimg = int(tick_kimg * MULTIPLIER)
    print(f"  1 Tick = {tick_kimg} kimg = {status_nimg} images")

    # 2. Total Duration
    duration_kimg = params['duration_kimg']
    duration_nimg = int(duration_kimg * MULTIPLIER)
    
    # 3. Snapshot Interval
    snap_ticks = params['snap_frequency']
    snapshot_nimg = int(snap_ticks * status_nimg)
    
    # 4. Checkpoint Interval
    dump_ticks = params['dump_frequency']
    checkpoint_nimg = int(dump_ticks * status_nimg)

    print(f"  Snapshot every {snap_ticks} ticks -> {snapshot_nimg} images")
    print(f"  Checkpoint every {dump_ticks} ticks -> {checkpoint_nimg} images")
    print("-" * 40)

    # --- Logic 3: Build Command ---
    command = [
        'torchrun',
        '--standalone',
        f"--nproc_per_node={params['gpus']}",
        str(train_script),
        
        # Data & Output
        f"--outdir={paths['output_dir']}",
        f"--data={paths['dataset_latent_zip']}", 
        
        # Architecture & Cond
        f"--preset={preset_name}",
        f"--cond={'1' if params['conditional'] else '0'}",
        
        # Hyperparameters
        f"--batch={params['batch_size']}",
        f"--batch-gpu={params['batch_gpu']}",
        
        # STRICT: Config MUST specify dropout (e.g. 0.10)
        f"--dropout={params['dropout']}", 
        
        # Runtime Settings (Using aligned NIMG values)
        f"--duration={duration_nimg}",     
        f"--snapshot={snapshot_nimg}",     
        f"--checkpoint={checkpoint_nimg}", 
        f"--status={status_nimg}",         
        
        # Performance
        f"--fp16={'1' if params['fp16'] else '0'}",
        f"--ls=1.0"
    ]

    # Resume Logic
    resume_path = paths.get('resume_pt_path')
    if resume_path:
        print(f"  [INFO] Resuming training from: {resume_path}")
        command.append(f'--resume={resume_path}')

    # --- Execution Logging ---
    print("="*80)
    print(f"LAUNCHING EDM2 LATENT DIFFUSION")
    print(f"  Preset:      {preset_name}")
    print(f"  Data:        {paths['dataset_latent_zip']}")
    print(f"  Dropout:     {params['dropout']}")
    print("-" * 80)
    print("Command:\n" + " ".join(command))
    print("="*80)

    try:
        Path(paths['output_dir']).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training Failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyError as e:
        print(f"\n[CONFIG ERROR] Missing required key in configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()