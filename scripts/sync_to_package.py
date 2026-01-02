#!/usr/bin/env python3
"""
Sync files from root directory to vllm_playground package directory.

This script copies files from the root (used for direct cloning/running) to the
package directory (used for PyPI distribution), applying necessary transformations.

Usage:
    python scripts/sync_to_package.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be synced without making changes
    --verbose   Show detailed output
"""

import os
import shutil
import argparse
import hashlib
from pathlib import Path
from datetime import datetime


# Project root directory
ROOT_DIR = Path(__file__).parent.parent
PKG_DIR = ROOT_DIR / "vllm_playground"


# Files/directories to sync (source relative to ROOT_DIR)
# Format: (source, destination_relative_to_PKG_DIR, transform_function_or_None)
SYNC_FILES = [
    # Python files with transformations
    ("app.py", "app.py", "transform_app_py"),
    ("container_manager.py", "container_manager.py", None),
    
    # Static files (direct copy)
    ("index.html", "index.html", None),
    ("benchmarks.json", "benchmarks.json", None),
    
    # Directories (recursive copy)
    ("static", "static", None),
    ("assets", "assets", None),
    ("config", "config", None),
    ("recipes", "recipes", None),
]

# Files/directories to skip in package (package-specific, don't overwrite)
SKIP_IN_PACKAGE = [
    "__init__.py",
    "cli.py",
    "__pycache__",
]


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file"""
    if not filepath.exists():
        return ""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def transform_app_py(content: str) -> str:
    """
    Transform app.py for package distribution.
    
    Changes:
    1. Add container_manager = None initialization
    2. Change to relative import for container_manager
    3. Add guards for container_manager usage
    """
    
    # Check if already transformed (has relative import)
    if "from .container_manager import" in content:
        print("  ⚠️  app.py already has package transforms, skipping transformation")
        return content
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Transform 1: Add None initialization before try block
        if line.strip() == "# Import container manager (optional - only needed for container mode)":
            new_lines.append(line)
            # Check if next line is 'try:' and add initialization
            if i + 1 < len(lines) and lines[i + 1].strip() == "try:":
                new_lines.append("container_manager = None  # Initialize as None for when import fails")
                new_lines.append(lines[i + 1])
                i += 2
                continue
        
        # Transform 2: Change to relative import
        elif "from container_manager import container_manager" in line:
            new_lines.append(line.replace(
                "from container_manager import container_manager",
                "from .container_manager import container_manager"
            ))
            i += 1
            continue
        
        # Transform 3: Add container_manager guards for common patterns
        # Pattern: if current_run_mode == "container":
        #          status = await container_manager.get_container_status()
        elif 'if current_run_mode == "container":' in line and "CONTAINER_MODE_AVAILABLE" not in line:
            # Check if next non-empty line uses container_manager
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            
            if next_idx < len(lines) and "container_manager." in lines[next_idx]:
                # Add guards
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + 'if current_run_mode == "container" and CONTAINER_MODE_AVAILABLE and container_manager:'
                new_lines.append(new_line)
                i += 1
                continue
        
        # Pattern: if CONTAINER_MODE_AVAILABLE: (without container_manager check)
        elif line.strip() == "if CONTAINER_MODE_AVAILABLE:" or \
             (line.strip().startswith("if CONTAINER_MODE_AVAILABLE") and "container_manager" not in line):
            # Check if next lines use container_manager
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            
            if next_idx < len(lines) and "container_manager." in lines[next_idx]:
                indent = len(line) - len(line.lstrip())
                new_line = line.rstrip().replace(
                    "if CONTAINER_MODE_AVAILABLE:",
                    "if CONTAINER_MODE_AVAILABLE and container_manager:"
                )
                new_lines.append(new_line)
                i += 1
                continue
        
        # Pattern: if current_run_mode == "container" and is_kubernetes:
        elif 'if current_run_mode == "container" and is_kubernetes:' in line and "container_manager" not in line:
            # Check if next lines use container_manager
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            
            if next_idx < len(lines) and "container_manager" in lines[next_idx]:
                new_line = line.replace(
                    'if current_run_mode == "container" and is_kubernetes:',
                    'if current_run_mode == "container" and is_kubernetes and container_manager:'
                )
                new_lines.append(new_line)
                i += 1
                continue
        
        # Pattern: if is_kubernetes and hasattr(container_manager
        elif "if is_kubernetes and hasattr(container_manager" in line and "and container_manager and" not in line:
            new_line = line.replace(
                "if is_kubernetes and hasattr(container_manager",
                "if is_kubernetes and container_manager and hasattr(container_manager"
            )
            new_lines.append(new_line)
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    # Additional transforms for robustness
    result = '\n'.join(new_lines)
    
    # Add guard to read_logs_container function if not present
    if "async def read_logs_container():" in result and "if not container_manager:" not in result:
        result = result.replace(
            'async def read_logs_container():\n    """Read logs from vLLM container"""\n    global vllm_running\n    \n    try:',
            'async def read_logs_container():\n    """Read logs from vLLM container"""\n    global vllm_running\n    \n    if not container_manager:\n        logger.error("read_logs_container called but container_manager is not available")\n        return\n    \n    try:'
        )
    
    # Fix container mode validation check
    if 'if config.run_mode == "container" and not CONTAINER_MODE_AVAILABLE:' in result:
        result = result.replace(
            'if config.run_mode == "container" and not CONTAINER_MODE_AVAILABLE:',
            'if config.run_mode == "container" and (not CONTAINER_MODE_AVAILABLE or not container_manager):'
        )
    
    return result


def sync_file(src: Path, dst: Path, transform_func=None, dry_run=False, verbose=False):
    """Sync a single file"""
    
    # Read source
    with open(src, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Apply transformation if specified
    if transform_func:
        transform = globals().get(transform_func)
        if transform:
            content = transform(content)
    
    # Check if destination exists and is different
    if dst.exists():
        with open(dst, 'r', encoding='utf-8', errors='replace') as f:
            existing = f.read()
        if existing == content:
            if verbose:
                print(f"  ⏭️  {src.name} (unchanged)")
            return False
    
    if dry_run:
        print(f"  📝 Would sync: {src} → {dst}")
        return True
    
    # Write to destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Synced: {src.name}")
    return True


def sync_directory(src: Path, dst: Path, dry_run=False, verbose=False):
    """Sync a directory recursively"""
    
    synced = 0
    
    if not src.exists():
        print(f"  ⚠️  Source directory not found: {src}")
        return 0
    
    for item in src.rglob('*'):
        if item.is_file():
            # Skip __pycache__ and other ignored patterns
            if any(skip in str(item) for skip in ['__pycache__', '.pyc', '.DS_Store']):
                continue
            
            rel_path = item.relative_to(src)
            dst_file = dst / rel_path
            
            # Check if file is different
            if dst_file.exists():
                if get_file_hash(item) == get_file_hash(dst_file):
                    if verbose:
                        print(f"  ⏭️  {rel_path} (unchanged)")
                    continue
            
            if dry_run:
                print(f"  📝 Would sync: {rel_path}")
                synced += 1
            else:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_file)
                if verbose:
                    print(f"  ✅ Synced: {rel_path}")
                synced += 1
    
    return synced


def main():
    parser = argparse.ArgumentParser(description="Sync files from root to package directory")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be synced")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 vLLM Playground - Sync to Package")
    print("=" * 60)
    print(f"Root:    {ROOT_DIR}")
    print(f"Package: {PKG_DIR}")
    if args.dry_run:
        print("Mode:    DRY RUN (no changes will be made)")
    print("=" * 60)
    print()
    
    total_synced = 0
    
    for src_rel, dst_rel, transform in SYNC_FILES:
        src = ROOT_DIR / src_rel
        dst = PKG_DIR / dst_rel
        
        if not src.exists():
            print(f"⚠️  Source not found: {src_rel}")
            continue
        
        print(f"📁 {src_rel}" + (f" (with transforms)" if transform else ""))
        
        if src.is_file():
            if sync_file(src, dst, transform, args.dry_run, args.verbose):
                total_synced += 1
        else:
            synced = sync_directory(src, dst, args.dry_run, args.verbose)
            if synced > 0:
                print(f"   ({synced} files)")
            total_synced += synced
    
    print()
    print("=" * 60)
    if args.dry_run:
        print(f"📊 Would sync {total_synced} file(s)")
    else:
        print(f"✅ Synced {total_synced} file(s)")
    print("=" * 60)
    
    if not args.dry_run and total_synced > 0:
        print()
        print("💡 Next steps:")
        print("   1. Review changes: git diff vllm_playground/")
        print("   2. Test locally: python -c 'from vllm_playground import app'")
        print("   3. Build package: python -m build")


if __name__ == "__main__":
    main()

