"""
Evaluation Orchestrator for EDM2.
Calculates FID (and other metrics) for snapshots found in the output directory.

Features:
- Idempotent: Skips already processed snapshots (checks CSV).
- Auto-Reference: Generates reference stats from RGB zip if missing.
- robust Parsing: Handles text vs JSON output formats from EDM2.
"""
import sys
import argparse
import re
import csv
import subprocess
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2] 
sys.path.append(str(project_root))

from utils.helpers import load_and_resolve_config

# --- Helper Functions ---
def extract_kimg(path_obj):
    """
    Extracts the kimg count from standard EDM2 filenames.
    Format: network-snapshot-0012345-0.050.pkl -> returns 12345
    """
    match = re.search(r'snapshot-(\d+)-', path_obj.name)
    return int(match.group(1)) if match else -1

def parse_fid_from_log(log_text):
    """
    Attempts to find the FID score using multiple regex patterns
    to account for different EDM2 version output formats.
    """
    # Pattern 1: Standard Text Output (e.g., "fid = 308.484")
    match = re.search(r'fid\s*=\s*([\d\.]+)', log_text)
    if match: return float(match.group(1))

    # Pattern 2: JSON Output (e.g., "fid50k_full": 308.484)
    match = re.search(r'"fid[^"]*":\s*([\d\.]+)', log_text)
    if match: return float(match.group(1))

    # Pattern 3: Legacy/Verbose (e.g., "fid50k_full = 308.484")
    match = re.search(r'fid50k_full\s*=\s*([\d\.]+)', log_text)
    if match: return float(match.group(1))

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load Config
    try:
        config = load_and_resolve_config(Path(args.config))
        paths = config['image_generation']['edm']['paths']
        eval_cfg = config['image_generation']['edm']['evaluation']
    except Exception as e:
        print(f"[CRITICAL] Config loading failed: {e}")
        sys.exit(1)
    
    # Resolve Paths
    rgb_zip = Path(paths['dataset_rgb_zip'])
    ref_stats = Path(paths['ref_stats_path'])
    model_dir = Path(paths['output_dir'])
    results_csv = Path(paths['metric_results_csv'])
    
    # Locate EDM2 internal script
    tool_dir = Path(__file__).parent 
    calc_script = tool_dir / "calculate_metrics.py"

    if not calc_script.exists():
        print(f"[CRITICAL] Could not find calculation script at {calc_script}")
        sys.exit(1)

    # This ensures 'evaluation/' exists before we try to write stats or CSVs
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    ref_stats.parent.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # STEP 1: Ensure Reference Statistics Exist
    # ==============================================================================
    print("="*80)
    print("STEP 1: Reference Statistics Check")
    if not ref_stats.exists():
        print(f"[INIT] Reference stats not found.")
        print(f"       Generating from: {rgb_zip}")
        
        cmd = [
            sys.executable, str(calc_script), 'ref',
            f"--data={rgb_zip}",
            f"--dest={ref_stats}"
        ]
        try:
            # We run this in the tool_dir to ensure it finds dnnlib
            subprocess.run(cmd, check=True, cwd=tool_dir)
            print("[INIT] Reference stats created successfully.")
        except subprocess.CalledProcessError:
            print("[CRITICAL] Failed to create reference stats. Check RGB zip path.")
            sys.exit(1)
    else:
        print(f"[INIT] Using existing stats: {ref_stats}")

    # ==============================================================================
    # STEP 2: Snapshot Discovery
    # ==============================================================================
    print("-" * 80)
    print("STEP 2: Snapshot Discovery")
    
    ema_tag = str(eval_cfg.get('ema_filter', '*'))
    if ema_tag == "*":
        search_pattern = "network-snapshot-*.pkl"
    else:
        search_pattern = f"network-snapshot-*-{ema_tag}.pkl"
        
    print(f"[SCAN] Directory: {model_dir}")
    print(f"[SCAN] Pattern:   {search_pattern}")
    
    all_pkls = list(model_dir.glob(search_pattern))

    # Sort by kimg
    all_pkls.sort(key=extract_kimg)
    
    if not all_pkls:
        print("[WARN] No snapshots found matching criteria.")
        sys.exit(0)

    # Strategy Filter
    if eval_cfg.get('strategy') == 'last':
        # Get the highest kimg value available
        max_kimg = extract_kimg(all_pkls[-1])
        target_pkls = [p for p in all_pkls if extract_kimg(p) == max_kimg]
        print(f"[PLAN] Strategy 'last' selected. Evaluating latest tick ({max_kimg} kimg).")
    else:
        target_pkls = all_pkls
        print(f"[PLAN] Strategy 'all' selected. Evaluating {len(target_pkls)} models.")

    # ==============================================================================
    # STEP 3: Metric Calculation Loop
    # ==============================================================================
    print("-" * 80)
    print("STEP 3: Metric Calculation")
    
    gpus = eval_cfg.get('gpus', 1)
    num_images = eval_cfg.get('num_images', 50000)
    batch_size = eval_cfg.get('batch_size', 64)
    metrics_list = eval_cfg.get('metrics', 'fid')

    # Load existing results to skip duplicates
    processed_files = set()
    if results_csv.exists():
        with open(results_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if row: processed_files.add(row[0]) # Filename column

    for i, pkl in enumerate(target_pkls):
        print(f"\n[{i+1}/{len(target_pkls)}] Processing: {pkl.name}")
        
        if pkl.name in processed_files:
            print(f"       -> Already in results CSV. Skipping.")
            continue
            
        print(f"       -> Launching torchrun ({gpus} GPUs)...")
        
        cmd = [
            'torchrun', '--standalone', f"--nproc_per_node={gpus}",
            str(calc_script), 'gen',
            f"--net={pkl}",
            f"--ref={ref_stats}",
            f"--batch={batch_size}",
            f"--seed=0",           
            f"--num={num_images}", 
            f"--metrics={metrics_list}"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, cwd=tool_dir, capture_output=True, text=True)
            output_log = result.stdout
            
            fid_score = parse_fid_from_log(output_log)
            
            if fid_score is not None:
                print(f"       -> [SUCCESS] FID: {fid_score}")
                
                # Append to CSV
                file_exists = results_csv.exists()
                with open(results_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['filename', 'kimg', 'fid', 'path', 'log'])
                    writer.writerow([
                        pkl.name, 
                        extract_kimg(pkl), 
                        fid_score, 
                        str(pkl),
                        output_log[-1000:]
                    ])
            else:
                print(f"       -> [WARN] Could not parse FID from output.")
                print("       -> Tail of output log:")
                print(output_log[-1000:]) 

        except subprocess.CalledProcessError as e:
            print(f"       -> [ERROR] Evaluation subprocess failed.")
            print("       -> STDERR:\n" + e.stderr)
            continue

    print("="*80)
    print(f"Evaluation complete. Results saved to:\n{results_csv}")

if __name__ == "__main__":
    main()