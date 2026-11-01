# utils/helpers.py
# -*- coding: utf-8 -*-
"""
Contains common utility functions for the project, such as configuration
file loading and path resolution. This allows for centralized logic that
can be reused across multiple scripts.
"""
import sys
import yaml
import re
from pathlib import Path
import os
import shutil
import pandas as pd


def safe_save_csv(df: pd.DataFrame, target_path: Path):
    """
    Saves a DataFrame to a CSV file atomically using a temporary file.
    This prevents file truncation/corruption if the process is killed 
    (e.g., by SLURM SIGKILL) during the write operation.
    """
    # 1. Create a temporary file path (e.g., metadata.csv.tmp)
    temp_path = target_path.with_suffix(target_path.suffix + '.tmp')
    
    try:
        # 2. Write the full dataframe to the temporary file
        df.to_csv(temp_path, index=False)
        
        # 3. Perform an atomic rename/replace.
        # This is a metadata-only swap on Linux/HPC filesystems.
        os.replace(temp_path, target_path)
        
    except Exception as e:
        # Cleanup temp file if something went wrong during the write
        if temp_path.exists():
            temp_path.unlink()
        raise e

def _resolve_paths(node, root_config: dict):
    """
    Recursively traverses the config, resolving path variables like '${section.key}'.
    """
    if isinstance(node, dict): return {k: _resolve_paths(v, root_config) for k, v in node.items()}
    if isinstance(node, list): return [_resolve_paths(i, root_config) for i in node]
    if isinstance(node, str):
        for match in re.findall(r'\$\{(.*?)\}', node):
            parts = match.split('.')
            value = root_config
            try:
                for part in parts: value = value[part]
                node = node.replace(f'${{{match}}}', str(value))
            except (KeyError, TypeError): print(f"[Config Warning] Could not resolve placeholder: ${{{match}}}", file=sys.stderr)
    return node

def load_and_resolve_config(config_path: Path, max_passes: int = 5) -> dict:
    """
    Loads a YAML config file and iteratively resolves all internal path variables.
    
    This handles nested variables (e.g., a path that references another path variable)
    by repeatedly applying the resolution logic until all variables are expanded.

    Args:
        config_path (Path): The path to the master YAML configuration file.
        max_passes (int): The maximum number of resolution passes to attempt,
                          preventing infinite loops.

    Returns:
        dict: The fully resolved configuration dictionary.
    """
    if not config_path.is_file(): 
        print(f"[FATAL ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Iteratively resolve paths to handle nested variables.
    # Each pass resolves one level of nesting. 5 passes is a safe number
    # for any reasonable configuration structure.
    for _ in range(max_passes):
        config = _resolve_paths(config, config)
        
    # As a final check, we can verify no unresolved variables remain, but for now
    # the iterative approach is robust enough.
    
    return config

def cleanup_dir(dir_path: str):
    """
    Safely and recursively deletes a directory.
    
    Args:
        dir_path (str): The absolute path to the directory to be removed.
    """
    print("=" * 80)
    print("Executing Step: Cleanup")
    print("-" * 80)
    print(f"Target for cleanup: {dir_path}")

    if os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"\nSUCCESS: Successfully removed intermediate directory.")
        except OSError as e:
            print(f"\nERROR: Could not remove directory {dir_path}: {e}", file=sys.stderr)
    else:
        print("\nINFO: Intermediate directory not found, skipping cleanup.")