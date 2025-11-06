#!/usr/bin/env python3
"""
Launcher script for vLLM Playground
"""
import sys
from pathlib import Path

# Add the parent directory to path to import vllm
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from app import main

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 vLLM Playground - Starting...")
    print("=" * 60)
    print("\nFeatures:")
    print("  ⚙️  Configure vLLM servers")
    print("  💬 Chat with your models")
    print("  📋 Real-time log streaming")
    print("  🎛️  Full server control")
    print("\nAccess the Playground at: http://localhost:7860")
    print("Press Ctrl+C to stop\n")
    print("=" * 60)
    
    main()

